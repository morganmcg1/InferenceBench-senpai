# SENPAI Research State

- **Updated:** 2026-05-28T18:20Z
- **Most recent direction from human researcher team:** No GitHub Issues. Run
  scope fixed by operator: Scenario A parity launch, RTX PRO 6000 shakedown,
  2h budget, 1 GPU shared, 2 logical students. Launch closes
  `2026-05-28T18:36:57Z`.

## Terminal winner (merged, final result for this launch)

**PR #160 — scen-a-fern/fern-fp8-weights — 1.8873x speedup — MERGED to `ib-20260528-scen-a-r1`**

vLLM `--quantization fp8` on-the-fly weight quantization. Full eval 128/128
success, MMLU-Pro ratio 0.9597 (pass, tight but valid), TTFT.p50 0.2325s, W&B
`bt2ckkzz`. BASELINE.md updated.

## Probes resolved this round

| PR | Student | Arm | Outcome | Final reading |
|---|---|---|---|---|
| #158 | scen-a-frieren | A0/A1 BF16 prefill tuning | Closed | BF16 ceiling ~1.276x; prefill structure is not a lever at burst concurrency 1 |
| #160 | scen-a-fern | B0/B1 FP8 weights + full eval | **MERGED** | 1.8873x winner; B1 tight prefill structure no-op over B0 |
| #168 | scen-a-fern | C0 FlashInfer + FP8 | Closed | failed_to_boot — `_sm_scale` AssertionError in vLLM FlashInfer wrapper |
| #169 | scen-a-frieren | D0 FP8 + enforce_eager | Closed | -4.1% TTFT vs winner; cudagraphs are net-positive at burst concurrency 1 |
| #171 | scen-a-fern | E0 FlashInfer + FP8 + enforce_eager (rescue) | Closed | Same `_sm_scale` assertion with cudagraphs off — wrapper-construction path itself is broken |
| #176 | scen-a-frieren | F0 V0 engine + FP8 | Closed | failed_to_boot — vLLM 0.11.0 hard-asserts V1 in OpenAI API server; V0 unreachable on this image |
| #177 | scen-a-fern | G0 `--max-num-batched-tokens=32768` | Closed (aborted) | gpu_slot.py raised `--min-remaining-s 480` to 900s floor; 641s remained → slot refused. Launcher preserved. Hypothesis untested. |

## Key findings (carry forward)

- **FP8 weights are the dominant Scenario A lever** on RTX PRO 6000 in this
  image. Going from BF16 (1.276x) to FP8 weight GEMM (1.887x) is a ~48%
  relative TTFT reduction.
- **Prefill structure tuning** (chunked-prefill toggle, max-model-len,
  max-num-batched-tokens) **does not move TTFT** at concurrency 1 with
  8192-token inputs — confirmed in both BF16 (PR #158 A0/A1) and FP8
  (PR #160 B0 vs B1). The bottleneck is the GEMM, not scheduling.
- **CUDA graphs are net-positive** at burst concurrency 1 with FP8 weights
  on this hardware (PR #169 D0: -4.1% TTFT when disabled). Future arms must
  keep cudagraphs enabled.
- **FlashInfer prefill is unreachable on vLLM 0.11.0 + this RTX PRO 6000
  image.** Both PR #168 (cudagraph on) and PR #171 (cudagraph off) hit
  `vllm/v1/attention/backends/flashinfer.py:972 assert decode_wrapper._sm_scale
  == self.scale` during `kernel_warmup`. The wrapper-construction path itself
  is broken; `--enforce-eager` is not a workaround.
- **FlashInfer sampler JIT** fails on this image (missing `curand.h`).
  Workaround: explicit `VLLM_USE_FLASHINFER_SAMPLER=0` and
  `VLLM_DISABLE_FLASHINFER_SAMPLING=1` inside every launcher. The
  `senpai/runtime_env.sh` defaults do not persist through the gpu_slot shell.
- **vLLM V0 engine is unreachable on this image** via the OpenAI entrypoint
  (vLLM 0.11.0 hard-asserts `envs.VLLM_USE_V1` in
  `vllm/entrypoints/openai/api_server.py:209`). Engine-version probes need to
  come from a different engine family or a vLLM downgrade.

## Infrastructure carry-overs for next launch

- **cublas link bug, cold FlashInfer JIT cache** (Frieren PR #169):
  `Fp8LinearOp` on SM_120 routes to `flashinfer_w8a8_scaled_mm` → JIT compile
  of `gemm.so` → link fails (`-lcublas`, `-lcublasLt`) because nvidia pip
  cublas ships only `libcublas.so.12`/`libcublasLt.so.12`. The PR #160 winner
  launcher and `runtime_env.sh` will crash on any fresh JIT cache. Frieren's
  workaround (symlink to unversioned `.so` + prepend `LIBRARY_PATH`) is
  embedded in `senpai/launchers/A/frieren-eager-fp8/arm_d0_fp8_eager.sh`.
  Frieren's PR #178 is in flight to bake this fix into `senpai/runtime_env.sh`
  upstream of every launcher; if it lands before cutoff it carries this fix
  into the merged advisor branch.
- **gpu_slot.py quick-mode min-remaining floor is 900s** (Fern PR #177):
  `senpai/gpu_slot.py:91` does `min_remaining_s = max(min_remaining_s,
  defaults["min_remaining_s"])`, and quick-mode's default is **900s (15 min)**.
  Any `--min-remaining-s` flag in the launcher block is silently raised to
  this floor. **Late-window quick probes need ≥15 min of slack at the moment
  of acquisition**, plus an additional ~3-5 min budget for assignment →
  student-pickup → workspace-prep → smoke-check before that. Future advisor
  scheduling must respect this hardcoded floor when assigning quick probes
  near a launch cutoff. Fern correctly aborted G0 rather than bypassing the
  gate.

## Time budget (as of 18:29Z)

- Cutoff: `2026-05-28T18:36:57Z` → **~8 min remaining**.
- GPU never reacquired after PR #176 — Fern's G0 (PR #177) was refused by
  the 900s quick-mode floor. GPU has been idle since 18:16:47Z.
- Only Frieren's PR #178 (non-GPU cublas-fix infra patch) is in flight.

## Next-round directions (priority-ordered)

1. **Patch `senpai/runtime_env.sh`** to bake the cublas symlink + `LIBRARY_PATH`
   fix in so FP8 launchers survive a cold FlashInfer JIT cache. This is an
   infra bug-fix PR, separate from any hypothesis arm. High priority — the
   current winner launcher is fragile to fresh-pod environments.
2. **Triton attention backend + FP8** (`VLLM_ATTENTION_BACKEND=TRITON_ATTN`):
   alt prefill path that bypasses both FlashAttention-2 and FlashInfer's
   broken wrapper. Worth a quick probe.
3. **`torch.compile` optimization level + FP8**: vLLM 0.11 supports compile
   modes that may further fuse prefill ops. Quick to probe.
4. **GPTQ or AWQ instead of naive FP8**: a quantization scheme with better
   quality preservation than naive FP8 weight-only could both pass MMLU-Pro
   more safely (current ratio 0.9597 is only 0.0097 above the τ=0.95 floor)
   and may maintain or improve speed.
5. **SGLang FP8 + Triton or TensorRT-LLM**: engine-family diversity. The
   natural path to engine-version diversity now that vLLM V0 is confirmed
   unreachable on this image.
6. **vLLM patch to relax the `_sm_scale` assertion** or vLLM version bump
   that fixes FlashInfer 0.4.x on SM_120 — both unlock the FlashInfer prefill
   direction that this launch confirmed is otherwise closed on this image.
7. **H100 leaderboard run**: repeat the winning FP8 recipe on H100 for
   leaderboard-comparable claim. The RTX PRO 6000 numbers here are shakedown
   only.
8. **Tertiary single-flag levers within vLLM 0.11.0 V1 + FP8**:
   `--max-num-batched-tokens=32768` (B1 only tested 16384 — does one-shot
   prefill of 8k prompt help?), `--max-num-seqs=1` (single-concurrency KV
   allocation), `--enable-prefix-caching` toggle, `--num-scheduler-steps>1`
   (multi-step decode dispatch).

## Final probe this round (resolved)

**PR #176 (CLOSED)** — Frieren arm F0: `VLLM_USE_V1=0` + FP8 weights →
**failed_to_boot.** vLLM 0.11.0 hard-asserts `envs.VLLM_USE_V1` at
`api_server.py:209`. V0 engine path is unreachable through the OpenAI
entrypoint on this image. Clean ruling-out — drop V0 engine probes from the
search surface for vLLM 0.11.0 + this image. Carry SGLang/TensorRT-LLM as
the path to engine-version diversity.

## Final Fern probe this round

**PR #178 (CREATED)** — Fern arm G0: FP8 weights +
`--max-num-batched-tokens=32768` (force one-shot prefill of the full 8k
prompt). Single CLI flag change on top of the PR #160 winner recipe, with
the cublas symlink workaround embedded for cold FlashInfer JIT cache safety.
B1 only tested 16384; pushing to 32768 ensures the entire 8192-token prompt
plus any chat-template tokens lands in one prefill chunk (vs the default
8192 max that may split across 2 chunks). Quick-only; terminal=false
regardless. The GPU is free, so no queue wait.
