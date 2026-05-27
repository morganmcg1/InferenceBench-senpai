# SENPAI Research State — ib-20260527-latest3-r1

- **As of:** 2026-05-27 15:30 UTC (55 min into 2h run, ~65 min remaining)
- **Last human directive:** none (no open GitHub Issues from research team)

## Research focus

InferenceBench LLM inference optimization. Goal: maximize per-scenario speedup
over PyTorch baseline on Mistral-7B-Instruct-v0.3 (RTX PRO 6000 shakedown, single
GPU, 2 hour budget) while passing the MMLU-Pro quality gate.

Reference table shows where the headroom lives:
- **A (input-heavy):** PyTorch 1.0x → vLLM default 1.25x → SMAC3 best 4.37x. Huge gap. Levers: chunked prefill, max-num-batched-tokens, prefix caching. On RTX PRO 6000 without FP8/FlashInfer, headroom appears limited to ~1.3-1.5x.
- **B (output-heavy):** PyTorch 1.0x → vLLM default 2.25x → SMAC3 best 15.23x. Biggest relative gap. **BREAKTHROUGH: ngram speculative hits 3.44x quick** — full B needs a dedicated launch (~65 min eval time).
- **C (high-load):** **MERGED WINNER: SGLang 22.48x** (PR #124). SGLang 0.5.9 with triton backend, mem-fraction-static 0.85, max-running-requests 128.
- **D (general):** PyTorch 1.0x → vLLM default 1.96x → SMAC3 best 5.69x. Balanced workload. ngram-spec from B transferring here now.

## Current portfolio (as of 15:30 UTC)

One confirmed winner merged. Two D-bound experiments racing for GPU:

1. **frieren → PR #122 → Scenario D pivot (vLLM ngram-spec)** — Arm 3 ngram-spec hit 3.44x quick on B. Full B doesn't fit (~65 min). Pivoting the same launcher to Scenario D (4096 in / 2048 out / c=4), full D ~26-35 min. Target: ≥2x terminal.
2. **fern → PR #123 → Scenario A wrap-up** — woke up late; Arm 1 quick = 1.273x. Full A doesn't fit (~73 min). Wrapping up with best quick as terminal-non-mergeable; notes FP8/FlashInfer unavailability as the limiting factor.
3. **tanjiro → PR #125 → Scenario D (vLLM ngram-spec)** — new assignment adapting frieren's B discovery to D. Racing frieren for GPU slot via gpu_slot.py FCFS.

## Coordination

- 1 GPU, 3 students. frieren and tanjiro both targeting D.
- Full D: ~26-35 min including quality gate.
- Hard cutoff: do not start full eval with < 35 min remaining (16:00 UTC).

## Potential next directions (future launches)

- **Scenario B full eval** — ngram-spec quick 3.44x on B is very promising; needs dedicated launch (65 min full eval time). Top priority for next run.
- **Scenario A on H100 with FP8** — RTX PRO 6000 without FP8/FlashInfer caps at ~1.3x on A. Real gains need H100 with FP8 KV cache.
- **SGLang for D** — not tested this run; good portfolio diversification for next launch.
- **Cross-scenario confirmation** — once A+B+D have terminal winners, compute aggregate geomean for leaderboard comparison.
- **Push C further** — current 22.48x; H100 reference shows SGLang default at 51.12x. More aggressive tuning (max-running-requests 256, LPM scheduler, torch compile) could push further.
