# InferenceBench SENPAI Live Baseline — ib-20260528-12h-r1

- **Research tag:** `ib-20260528-12h-r1`
- **Advisor branch:** `ib-20260528-12h-r1`
- **Target base branch:** `codex/inferencebench-senpai-target`
- **Hardware (this launch):** NVIDIA RTX PRO 6000 Blackwell-class, ~96GB VRAM (shakedown, NOT leaderboard-comparable to H100)
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Quality gate:** MMLU-Pro 500-sample subset (seed 248), `tau = 0.95` of PyTorch baseline (0.298)
- **W&B project:** `wandb-applied-ai-team/inferencebench-senpai`
- **Scoring assets:** imported from `rtxpro6000-seed248` PVC snapshot, preflight passed 2026-05-28
- **Time budget:** 720 minutes (12h) — `SENPAI_TIMEOUT_MINUTES=720`

## Current best per scenario (live advisor state)

| Scenario | Primary metric | Current best speedup | Launcher | W&B run | PR | Notes |
|---|---|---:|---|---|---|---|
| A: Input-heavy | `scenario/A/speedup_over_pytorch` (1/ttft.p50, burst c=1, 128 reqs) | — | — | — | — | unmeasured on this hardware |
| B: Output-heavy | `scenario/B/speedup_over_pytorch` (1/tpot.p50, burst c=1, 64 reqs) | — | — | — | — | unmeasured on this hardware |
| C: High-load | `scenario/C/speedup_over_pytorch` (geomean req/s across burst c=64, poisson 32, constant 16) | — | — | — | — | unmeasured on this hardware |
| D: General | `scenario/D/speedup_over_pytorch` (geomean of 1/ttft, 1/tpot, req/s, burst c=4, 96 reqs) | — | — | — | — | unmeasured on this hardware |
| Aggregate | `aggregate/geomean_speedup_over_pytorch` (geomean of A–D) | — | — | — | — | confirmed only after A–D each have a clean full result |

PyTorch baseline raw objectives (seed 248, RTX PRO 6000, this launch) live in the PVC-imported assets and are loaded from `$INFERENCE_BENCH_PYTORCH_BASELINE_METRICS` automatically by `senpai/summarize_metrics.py` / `validate_result.py`.

## Reference snapshot (H100, 2h, public — NOT this hardware)

From `program.md` 2026-05-21 snapshot for context only. RTX PRO 6000 results MUST NOT be compared 1:1 against these.

| Method | Aggregate | A TTFT | B TPOT | C req/s | D geomean |
|---|---:|---:|---:|---:|---:|
| SMAC3 search, 2h vLLM | 11.53x | 4.37x | 15.23x | 46.70x | 5.69x |
| TPE search, 2h vLLM | 11.25x | 4.48x | 14.76x | 43.46x | 5.58x |
| Random search, 2h vLLM | 10.20x | 4.21x | 11.34x | 41.81x | 5.42x |
| vLLM default, no agent | 4.05x | 1.25x | 2.25x | 48.69x | 1.96x |
| SGLang default, no agent | 3.92x | 1.22x | 1.77x | 51.12x | 2.14x |

## Update history

- 2026-05-28 — initial ledger created. Preflight passed against PVC assets `rtxpro6000-seed248`. Three students assigned to A/B/C diversification round.

## Rules

- Update the "current best" row only after `senpai/validate_result.py` prints `validation_pass=true` and `baseline_update_allowed=true` against a full clean-relaunch `metrics_full.json` tied to a PR on this advisor branch.
- Quick-only probes, partial requests, raw-objective values, or runs without W&B IDs stay out of the current-best row. Track them in PR comments and the experiments log instead.
- Quality gate failure invalidates the speed result regardless of magnitude.
