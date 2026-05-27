# SENPAI Research State — ib-20260527-latest3-r1

- **As of:** 2026-05-27 15:58 UTC (83 min into 2h run, ~37 min until 16:35 review cutoff)
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

## Current portfolio (as of 15:58 UTC)

One confirmed winner merged. Three terminal-stage nudges out at 15:57 UTC:

1. **frieren → PR #122** — silent since 15:36. Pivot to D was instructed but no W&B group `frieren-vllm-d-ngram` exists yet. GPU shows 96% util / 87GB used (likely a stale loaded server). Asked to either salvage with terminal SENPAI-RESULT (B 3.44x quick, non-mergeable) or post the D quick if pivoted.
2. **fern → PR #123** — silent since 15:15 (Arm 1 quick 1.273x). Asked to commit launcher and post terminal-non-mergeable now; no Arm 2, no full A.
3. **tanjiro → PR #125** — claimed GPU lease was held by frieren until 16:44, but `gpu_slot.py status` shows no active lease (slot stale). Asked to retry the GPU acquisition with a short min-remaining-s (10 min). If quick D >=2x AND >=30 min remain, run full D; else commit launcher + terminal-non-mergeable quick.

Latest W&B activity: nothing logged since 15:26 (tanjiro's full C heartbeat). No D group has emerged for either frieren or tanjiro.

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
