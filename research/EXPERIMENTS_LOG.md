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
