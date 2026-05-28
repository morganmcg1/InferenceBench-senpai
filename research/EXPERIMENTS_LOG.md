# SENPAI Research Results — ib-20260528-scen-a-r1

## 2026-05-28 17:20 — PR #158 / PR #160 partial quick-probe summary (non-terminal)

Both Scenario A arms posted quick `SENPAI-RESULT` checkpoints; neither has a
terminal full-eval result yet. Recording the quick frontier so we have it on
hand if the run window closes before the full evals finish.

### PR #158 — `scen-a-frieren/frieren-prefill-tuning` (BF16 prefill tuning)

| Arm | Launcher | TTFT.p50 (s) | Speedup vs PyTorch | success/total | W&B run | Eval mode |
|---|---|---:|---:|---:|---|---|
| A0 default `vllm_running` | `src/starting_points/vllm_running/start_server.sh` | 0.3436 | 1.276x | 4/4 | `e167f5c6` | quick |
| A1 chunked-prefill off + tight | `senpai/launchers/A/frieren-prefill-tuning/arm_a1_chunkedoff_bt16k.sh` | 0.3435 | 1.277x | 4/4 | `apt7l3yj` | quick |

- BF16 weights, FlashAttention, FlashInfer prefill disabled (runtime_env default).
- A0 used `--max-model-len 32768 --gpu-memory-utilization 0.90` (runtime_env default for `MAX_MODEL_LEN`).
- A1 used `--max-model-len 9728 --max-num-batched-tokens 16384 --max-num-seqs 8 --gpu-memory-utilization 0.92 --no-enable-chunked-prefill --no-enable-prefix-caching`.
- VRAM peak ~89.3 GB on A0.
- Quality screening at n=16: 0.25 / 0.298 (below 0.95 ratio floor, but this is `quality_evidence=screening_only` noise at n=16; full quality gate is the actual check).

**Conclusion:** At burst concurrency 1 with an ~8192-token prompt, vLLM's
default chunked-prefill already produces one prefill kernel call; toggling
chunked-prefill off does not change TTFT. Tightening `max-model-len` and
`max-num-batched-tokens` also does not move TTFT here because the bottleneck
is the prompt-token GEMM, not warmup or KV-cache allocation.

**Advisor steering:** Skip A2 control (foregone). Promote A0 to full eval only
if wall-time allows; otherwise document the BF16 ceiling and submit
terminal=false.

### PR #160 — `scen-a-fern/fern-fp8-weights` (FP8 weight quantization)

| Arm | Launcher | TTFT.p50 (s) | Speedup vs PyTorch | success/total | W&B run | Eval mode |
|---|---|---:|---:|---:|---|---|
| B0 FP8 default | `senpai/launchers/A/fern-fp8-weights/arm_b0_fp8_default.sh` | 0.2313 | **1.896x** | 4/4 | `gr5ge1fd` | quick |
| B1 FP8 + tight prefill | `senpai/launchers/A/fern-fp8-weights/arm_b1_fp8_tight.sh` | 0.2301 | **1.906x** | 4/4 | `ug454rhd` | quick |

- vLLM `--quantization fp8` on-the-fly quantization of the assigned
  `mistralai/Mistral-7B-Instruct-v0.3` checkpoint; KV cache stays BF16
  (FlashAttention rejects FP8 KV per program.md).
- B0: `--gpu-memory-utilization 0.90` only.
- B1: adds `--max-model-len 9728 --max-num-batched-tokens 16384 --max-num-seqs 8 --gpu-memory-utilization 0.92 --no-enable-chunked-prefill --no-enable-prefix-caching`.
- VRAM peak ~89-90 GB.
- Quality is `screening_only` at n=16 (ratio 0.84 — noise). Full quality gate
  is the actual check; primary risk on this PR.

**Conclusion:** FP8 weight quantization is the dominant Scenario A lever on
this hardware. B1's tight prefill structure does not improve over B0 default
(0.5% delta, below 5% promotion floor) — consistent with Frieren's BF16
finding that prefill structure is not the bottleneck at concurrency 1.

**Advisor steering:** Decision tree confirmed; promote B0 (the simpler recipe)
to full eval immediately. Critical that MMLU-Pro full quality gate (n=500)
passes at ratio ≥ 0.95.

### Infrastructure note (caught by Fern)

`senpai/runtime_env.sh` sets `VLLM_USE_FLASHINFER_SAMPLER=0` and
`VLLM_DISABLE_FLASHINFER_PREFILL=1`, but those env vars do not stick through
the shell that `gpu_slot.py run` spawns in this packed pod. vLLM 0.11.0's V1
sampler then tries to JIT-compile FlashInfer's `top_k_top_p_sampling_from_logits`
kernel and fails because `curand.h` is not on the nvcc include path. The
fix is to set both exports explicitly inside each launcher script. No changes
to `runtime_env.sh` were required. Frieren'\''s launcher should adopt the same
explicit exports.

### Standing follow-ups for next round

1. **Combined FP8 + best BF16-style knob recipe**, if any BF16-side knob
   surfaces an additional win during the FP8 full eval (none expected from
   the current data, but worth a quick probe).
2. **FP8 + `--max-num-seqs=1`** to probe whether KV allocation tightening
   helps at single-concurrency.
3. **FlashInfer prefill controlled probe** as an explicit test path with the
   default sampler disabled.
4. **Engine-family probe** (SGLang Triton backend) — only if vLLM hits a
   prefill ceiling that FP8 + structure cannot push past.

## 2026-05-28 17:34 — PR #160 (MERGED, current best): FP8 weight quantization full eval

- `scen-a-fern/fern-fp8-weights`
- **Hypothesis:** On-the-fly vLLM `--quantization fp8` halves prefill GEMM bandwidth,
  yielding large TTFT.p50 reduction at burst concurrency 1 while preserving
  MMLU-Pro quality above the τ=0.95 floor.
- **Result (full eval, 128/128, seed 248):**

| Metric | Value |
|---|---:|
| `scenario/A/speedup_over_pytorch` | **1.8873x** |
| `scenario/A/inverse_ttft_p50` (raw) | 4.3038 req/s |
| TTFT.p50 (s) | 0.2325 |
| MMLU-Pro observed accuracy (n=500) | 0.286 |
| MMLU-Pro ratio (τ=0.95 floor) | 0.9597 (pass) |
| Speed success / total | 128 / 128 |
| VRAM peak | ~89.4 GB |
| W&B run | `bt2ckkzz` (group `fern-fp8-weights`) |

- **Conclusion:** Validated terminal winner. FP8 weight quantization is the
  dominant Scenario A lever on RTX PRO 6000 in this image. Merged to
  `ib-20260528-scen-a-r1`; `BASELINE.md` updated. Quality is just above the
  0.95 floor (ratio 0.9597) — any subsequent FP8 variant that further reduces
  model fidelity risks pushing quality below the gate.

## 2026-05-28 17:55 — PR #168 (CLOSED): FlashInfer prefill + FP8 (quick probes)

- `scen-a-fern/fern-flashinfer-fp8`
- **Hypothesis:** FlashInfer's paged-attention prefill kernel (vLLM
  `VLLM_ATTENTION_BACKEND=FLASHINFER` with `VLLM_DISABLE_FLASHINFER_PREFILL=0`)
  is ~15-30% faster than FlashAttention-2 at 8192-token prefill on modern CUDA
  GPUs; combined with FP8 weights it could push TTFT.p50 below 0.197s.
- **Result:** **failed_to_boot.** vLLM crashes during cudagraph warmup with
  `vllm/v1/attention/backends/flashinfer.py:972 AssertionError:
  decode_wrapper._sm_scale == self.scale`. Sampler kept disabled per
  fern-fp8-weights pattern; sampler JIT (`curand.h` missing) is a different
  failure mode. C1 (FP8 KV) was skipped per decision tree.
- **Conclusion:** FlashInfer prefill on vLLM 0.11.0 + this RTX PRO 6000 image
  is unreachable with the in-tree backend. `VLLM_DISABLE_FLASHINFER_PREFILL=1`
  default in `runtime_env.sh` is correct.

## 2026-05-28 18:02 — PR #169 (CLOSED): FP8 + --enforce-eager sensitivity (quick probe)

- `scen-a-frieren/frieren-eager-fp8`
- **Hypothesis:** At burst concurrency 1, CUDA graph dispatch overhead could
  be net-negative once the FP8 GEMM dominates; `--enforce-eager` would expose
  this if so.
- **Result (quick probe):**

| Metric | D0 | PR #160 winner | Δ |
|---|---:|---:|---:|
| `scenario/A/speedup_over_pytorch` | 1.8108x | 1.8873x | **-4.1%** |
| `scenario/A/inverse_ttft_p50` (raw) | 4.1293 req/s | 4.3038 req/s | -4.1% |
| TTFT.p50 (s) | 0.2422 | 0.2325 | +9.7 ms |
| W&B run | `d3oxwdko` (group `frieren-eager-fp8`) | `bt2ckkzz` | — |

- **Conclusion:** Hypothesis rejected. CUDA graphs are net-positive at burst
  concurrency 1 with FP8 weights on this image. Future Scenario A levers must
  keep cudagraphs enabled.
- **Infrastructure finding (carry to next round):** On a cold FlashInfer JIT
  cache the FP8 path crashes — vLLM's `Fp8LinearOp` on SM_120 routes to
  `flashinfer_w8a8_scaled_mm` which JIT-compiles `gemm.so`, and the link
  fails because the nvidia pip cublas package only ships versioned
  `libcublas.so.12`/`libcublasLt.so.12`. Frieren's launcher created
  `/tmp/inferencebench-cublas-stubs/lib{cublas,cublasLt}.so` symlinks and
  prepended that dir to `LIBRARY_PATH`. The PR #160 winner and
  `senpai/runtime_env.sh` will hit the same crash on a cold cache. Next-round
  infra task: bake the symlink fix into `senpai/runtime_env.sh` so every FP8
  launcher survives a cold JIT cache.

## 2026-05-28 18:04 — PR #171 (CLOSED): FlashInfer prefill + FP8 + --enforce-eager rescue

- `scen-a-fern/fern-flashinfer-eager`
- **Hypothesis:** PR #168 crashed during cudagraph warmup. If the
  `_sm_scale` assertion is cudagraph-specific, `--enforce-eager` rescues it.
- **Result:** **failed_to_boot, same assertion.** The probe confirmed
  `cudagraph_mode=0`, `max_capture_size=0`, `cudagraph_capture_sizes=[]` — yet
  the AssertionError fires inside `FlashInferImpl.forward` during
  `kernel_warmup._dummy_run` before any cudagraph would be captured. The
  wrapper-construction path itself is the broken layer, not cudagraph capture.
- **Conclusion:** FlashInfer prefill direction is closed for this image. To
  pursue this lever in a future launch: either (a) vLLM patch relaxing the
  assertion, (b) a vLLM version bump with known-good FlashInfer 0.4.x on
  SM_120, or (c) a different engine family (SGLang FP8, TensorRT-LLM) that
  bypasses vLLM's FlashInfer wrapper entirely.
