# SENPAI Research State

- **Updated:** 2026-05-28T16:45Z
- **Most recent direction from human researcher team:** No active GitHub Issues
  for this launch. Run scope is fixed by operator instructions: Scenario A
  parity launch on RTX PRO 6000 shakedown, 2h budget, 1 GPU shared between two
  students, paper-grade serving recipes only.

## Current research focus

Optimize `scenario/A/speedup_over_pytorch` (long-context prefill TTFT.p50 under
burst, concurrency 1, 8192-token input, 1024-token output). PyTorch baseline
TTFT.p50 on this hardware is 0.4385s; any clean-relaunch full-eval result that
beats this **and** passes the MMLU-Pro τ=0.95 quality gate is a candidate for
the live BASELINE.md current-best row.

The two open assignments cover the two most-likely-to-pay-off orthogonal
levers:

| Lever family | Student | Arm summary | W&B group |
|---|---|---|---|
| Prefill structure (BF16) | scen-a-frieren | A0 default vLLM baseline → A1 chunked-prefill off + tight `max-model-len` + `max-num-batched-tokens=16384` → A2 chunked on + 8192 control → full eval on best | `frieren-prefill-tuning` |
| Weight precision (FP8) | scen-a-fern | B0 vLLM `--quantization fp8` default → B1 FP8 + tight `max-model-len` + batched-tokens=16384 + chunked off → full eval; fallback plan F if FP8 boot/quality fails | `fern-fp8-weights` |

These were chosen because:

- FP8 weights and prefill scheduler tuning are roughly orthogonal: if both win
  on their own, the strongest result for the next round will combine FP8 with
  the best prefill structure.
- Both are vLLM-only. Engine-family diversity (SGLang/TGI/TensorRT-LLM) is a
  next-round option if these probes are inconclusive or if quick measurements
  expose vLLM-specific bottlenecks.
- Neither requires touching protected benchmark files; recipes live under
  `senpai/launchers/A/<student-slug>/`.

## Potential next research directions

Triggered by results from the current assignments:

1. **FP8 weights × prefill structure (combined recipe).** If both arms beat
   baseline independently, the obvious next PR is the combined launcher (FP8 +
   chunked-prefill off + `max-num-batched-tokens 16384` + tight max-model-len).
2. **Engine-family probe.** A SGLang Scenario-A launcher (Triton attention
   backend, `--chunked-prefill-size` swept) as a non-vLLM control, especially
   useful if vLLM hits a prefill kernel ceiling.
3. **CUDA-graph / compile sensitivity at concurrency 1.** `--enforce-eager`
   true vs false on the best BF16 recipe to attribute CUDA-graph value when
   only one sequence is live. Often surprising at concurrency 1.
4. **`max-num-seqs=1` sensitivity.** vLLM still allocates KV blocks for
   `max-num-seqs`; tightening to 1 may free more KV headroom on RTX PRO 6000.
5. **FlashInfer prefill controlled probe.** A single quick arm that flips
   `VLLM_DISABLE_FLASHINFER_PREFILL=0`, only after the runtime is otherwise
   clean. The runtime_env helper disables it by default; this is the explicit
   test path called out in `program.md`.
6. **Speculative decoding for the 1024-token tail of A.** A speculation arm
   wouldn't reduce TTFT (that's pure prefill), but if A's primary metric
   measurement window blends prefill+decode, n-gram speculation could still
   matter. Verify by reading `runner.py` before assigning.

## Constraints reminder

- Launch deadline: `2026-05-28T18:36:57Z` (hard cutoff via cluster gate).
- Reserve last 10-15 minutes purely for advisor review, merge, and BASELINE.md.
- Treat full eval as 30-45 min + 5-10 min cold restart on shared GPU; do not
  start a full eval without `--min-remaining-s 1800`.
- RTX PRO 6000 shakedown results are not H100 leaderboard-comparable; flag
  every result with the hardware so future-self isn't fooled.
