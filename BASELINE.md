# InferenceBench SENPAI Baseline — ib-20260522-r2

- Last updated: 2026-05-22
- Base model: `mistralai/Mistral-7B-Instruct-v0.3`
- Hardware: 1x NVIDIA H100 80GB
- Time budget: 2 hours per scenario (InferenceBench protocol)
- Starting launcher: `src/starting_points/vllm_running/start_server.sh` (vLLM default with `--gpu-memory-utilization 0.90`, no other tuning)
- Reference snapshot source: `target/program.md` (2026-05-21 public InferenceBench leaderboard)

## Current best per scenario

| Scenario | Primary metric | Current best speedup | Source | Notes |
|---|---|---:|---|---|
| A: Input-heavy | `scenario/A/speedup_over_pytorch` (1/ttft.p50, burst) | 4.37x | Reference: SMAC3 2h vLLM | No SENPAI run validated yet |
| B: Output-heavy | `scenario/B/speedup_over_pytorch` (1/tpot.p50, burst) | 15.23x | Reference: SMAC3 2h vLLM | No SENPAI run validated yet |
| C: High-load | `scenario/C/speedup_over_pytorch` (geomean req/s across burst/poisson/constant) | 46.70x | Reference: SMAC3 2h vLLM | No SENPAI run validated yet |
| D: General | `scenario/D/speedup_over_pytorch` (geomean of 1/ttft, 1/tpot, req/s, burst) | 5.69x | Reference: SMAC3 2h vLLM | No SENPAI run validated yet |
| Aggregate | `aggregate/geomean_speedup_over_pytorch` | 11.53x | Reference: SMAC3 2h vLLM | Mature-winner cross-scenario only |

## SENPAI-validated launcher recipes

(none yet — first round of experiments dispatched 2026-05-22)

## Update history

- 2026-05-22 — initial baseline populated from public reference snapshot in `target/program.md`. Round 1 of experiments dispatched to r2-frieren (B), r2-fern (A), r2-tanjiro (C).
