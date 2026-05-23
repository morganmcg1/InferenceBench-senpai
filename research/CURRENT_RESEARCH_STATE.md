# SENPAI Research State

- **Date / time:** 2026-05-23 11:08 UTC (mid-launch)
- **Most recent human researcher directive:** none — no open issues addressed to this advisor branch.
- **Active launch:** `ib-20260523-rerun-r2` on `ib-20260523-rerun-r2-advisor`
- **Hardware (this launch):** 1× RTX PRO 6000 (~96 GB), shakedown mode — not leaderboard-comparable.
- **Wall-clock budget:** 2 h total. Launch started 09:58 UTC. **~50 min remaining as of 11:08 UTC.**
- **Students:** r2-frieren, r2-fern, r2-tanjiro (3 logical students sharing 1 GPU pod).

## Current Research Focus

InferenceBench LLM-inference-serving optimization. Headline metric per scenario
is `speedup_over_pytorch` on a foreground OpenAI-compatible `start_server.sh`
launcher; quality gate is a strict prerequisite (MMLU-Pro ratio ≥ 0.95 of
PyTorch baseline).

We are in shakedown mode on RTX PRO 6000. The first priority remains making
any terminal `SENPAI-RESULT` legitimate (real `speedup_over_pytorch` from a
matching torch baseline, not extrapolation from public H100 ratios).

### Slot status (11:08 UTC)

- **Slot 1 — r2-fern (Scenario A):** RAN 09:58 → 10:51 UTC. Pre-empted r2-frieren
  for the GPU. Posted **partial** SENPAI-RESULT at 10:54: no Scenario A torch
  baseline, no MMLU-Pro quality run, vLLM full eval killed at ~73% before
  writing `metrics_full.json`. `metrics_quick.json` and the long-prefill
  launcher are committed at `senpai/launchers/scenario_a/vllm_long_prefill/`.
  Not a winner. PR #29 left open as documented partial.
- **Slot 2 — r2-frieren (bootstrap + Scenario B):** Committed CPU-only bootstrap
  at 10:52 — `senpai/materialize_requests.py` patch plus the four
  `src/eval/inference/baselines/speed/torch/<scenario>/.../requests.jsonl`
  files. **Authorized to take the GPU at 11:07 UTC** for Scenario B torch
  baseline + vLLM eval, ~35 min budget. Launcher committed at
  `senpai/launchers/scenario_b/vllm_fp8_kv_flashinfer/start_server.sh`.
- **Slot 3 — r2-tanjiro (Scenario D):** Holding. Scenario D launcher already
  committed at `senpai/launchers/scenario_d/vllm_fp8_balanced/start_server.sh`.
  Will take the slot after r2-frieren posts `SLOT-FREE` — ~15–18 min expected,
  using `--request-limit 16` for baseline + vLLM eval.

### Scenario priorities for this launch

1. **Scenario B (output-heavy, TPOT)** — biggest known headroom (vLLM default
   2.25× vs SOTA 15.23× on H100). FP8 KV cache, CUDA graphs, FlashInfer
   decode are the high-leverage levers. **In flight via r2-frieren now.**
2. **Scenario D (general)** — moderate headroom (1.96× → 5.69×). FP8 weight
   quant + FP8 KV + chunked prefill + FLASHINFER. **Queued via r2-tanjiro.**
3. **Scenario A (input-heavy, TTFT)** — long-prefill latency. The partial
   r2-fern run produced a recipe but not a speedup number. **No winner.**
4. **Scenario C (high-load)** — deprioritized; vLLM defaults already near or
   beating non-agentic SOTA at 48.69× vs SMAC3 46.70×. Not attempted this launch.

### Decisions that shaped the launch

- Original plan was frieren-first to bootstrap torch baselines, then fern,
  then tanjiro. r2-fern grabbed the GPU first (~10:00 UTC) before r2-frieren
  finished bootstrap, which pushed all three later. Recoverable because
  r2-frieren's deferred work was CPU-only (materialize_requests, launcher
  authoring), but cost the launch its torch Scenario A baseline.
- The original PR-body precompute_baseline CLI was wrong; r2-tanjiro caught
  it. Corrected workflow posted to all three PRs. The real CLI is
  `precompute_baseline --scenario-id <full_folder> --server-url <url> ...`
  against a separately-launched `transformers_openai_server.py`.
- Time pressure forced `--request-limit` caps (24 for B, 16 for D) instead
  of the full 64/96-request baseline. p50 readings are still meaningful; p90/p99
  are noisier and should be reported as such.

## Potential Next Research Directions (post-launch)

- **Re-run Scenario A on RTX PRO 6000** with a real torch baseline; the
  long-prefill launcher recipe from r2-fern is already committed and known
  to boot cleanly.
- **FP8 weight + FP8 KV combined** on B and D if the quality gate holds in
  this launch. Strong multiplicative-win candidate.
- **Speculative decoding (n-gram or draft model)** for B. vLLM search space
  lists `num_speculative_tokens ∈ {3, 5, 7}` but interacts with kernel and
  quant choices — needs deliberate A/B.
- **SGLang launcher alternative** for B (output-heavy decode is where SGLang's
  RadixAttention + zero-overhead scheduler may shine). The SGLang search
  space file currently disables most useful knobs — verify manually.
- **Cross-scenario confirmation** geomean run once any single-scenario PR
  becomes a clear winner.

This document is a living plan — prune and rewrite after every review cycle.
