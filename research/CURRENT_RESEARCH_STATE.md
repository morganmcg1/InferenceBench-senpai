# SENPAI Research State

- **Date / time:** 2026-05-23 (launch boot)
- **Most recent human researcher directive:** none — no open issues addressed to this advisor branch.
- **Active launch:** `ib-20260523-rerun-r2` on `ib-20260523-rerun-r2-advisor`
- **Hardware (this launch):** 1× RTX PRO 6000 (~96 GB), shakedown mode — not leaderboard-comparable.
- **Wall-clock budget:** 2 hours total for the whole research programme this launch.
- **Students:** r2-frieren, r2-fern, r2-tanjiro (3 logical students sharing 1 GPU pod).

## Current Research Focus

InferenceBench LLM-inference-serving optimization. The headline metric is
per-scenario `speedup_over_pytorch` on a foreground OpenAI-compatible
`start_server.sh` launcher; quality gate is a strict prerequisite (MMLU-Pro
ratio ≥ 0.95 of PyTorch baseline).

We are in shakedown mode on RTX PRO 6000. The first priority is to make any
terminal `SENPAI-RESULT` legitimate: establish the PyTorch speed baseline on
this exact hardware/model/seed so `speedup_over_pytorch` is real and not an
extrapolation from public H100 ratios. Once that is done, push optimized
launchers for the highest-leverage scenarios.

### Scenario priorities for this launch

1. **Scenario B (output-heavy, TPOT)** — biggest known headroom (vLLM default
   2.25× vs SOTA 15.23× on H100). FP8 KV cache, CUDA graphs, FlashInfer
   decode kernels, and FP8 weight quantization are the high-leverage levers.
2. **Scenario A (input-heavy, TTFT)** — long-prefill latency. Chunked prefill
   size, max-num-batched-tokens, FlashAttention prefill backend, prefix caching
   off (unique prompts) are the levers.
3. **Scenario D (general)** — only if A and B finish in time.
4. **Scenario C (high-load)** — deprioritized; vLLM defaults already near or
   beating non-agentic SOTA at 48.69× vs SMAC3 46.70×. Touch only if there is
   leftover GPU time after A/B/D.

## Coordination Plan (1 GPU, 3 students)

Heavy GPU work must be serialized. Pattern for this launch:

- **Slot 1 — r2-frieren (bootstrap + Scenario B):** runs preflight, materializes
  request files if absent, runs the PyTorch speed baseline for Scenario B,
  publishes that as the first authoritative `pytorch_baseline_metrics.json` for
  this hardware, then runs a tuned vLLM Scenario B launcher and posts
  `SLOT-FREE` when the GPU is fully released.
- **Slot 2 — r2-fern (Scenario A):** prepares launcher recipe and smoke
  command while waiting; once SLOT-FREE arrives, runs the PyTorch baseline for
  Scenario A and then the tuned vLLM Scenario A launcher. Posts `SLOT-FREE`.
- **Slot 3 — r2-tanjiro (Scenario D):** prepares launcher recipe and smoke
  command while waiting; once Slot 2 is free, runs PyTorch baseline for
  Scenario D and the tuned launcher. Posts `SLOT-FREE`.

Students should `pkill` only their own server process group, never broad
patterns. Every PR explicitly states whose GPU slot it is.

## Potential Next Research Directions

If the first round finishes early or yields strong winners that need
follow-up:

- **FP8 weight quantization + FP8 KV cache** as a single combined PR. Should
  give multiplicative wins on B and D if quality holds.
- **Speculative decoding (n-gram or draft model)** for B. The vLLM search
  space already lists `num_speculative_tokens ∈ {3,5,7}` but it interacts
  delicately with kernel and quant choices.
- **SGLang launcher alternative** for B (output-heavy decode is where SGLang's
  RadixAttention + zero-overhead scheduler can shine), but the SGLang search
  space file currently disables most of its useful knobs — verify them
  manually before assigning.
- **Custom TensorRT-LLM** launcher for A (long-context prefill on Blackwell)
  if a winner is needed beyond vLLM tuning.
- **Cross-scenario confirmation** geomean run once any single-scenario PR
  becomes a clear winner.

This document is a living plan — prune and rewrite after every review cycle.
