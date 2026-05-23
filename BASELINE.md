# SENPAI InferenceBench Baseline Ledger

Live advisor-owned baseline ledger for research tag `ib-20260523-rerun-r1` on
advisor branch `ib-20260523-rerun-r1-advisor`.

## Run Context

- **Date launched:** 2026-05-23
- **Hardware (shakedown):** 1 x NVIDIA RTX PRO 6000 Blackwell (~96 GB VRAM)
- **Model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Time budget:** 2 hours wall clock for the whole research program
- **Reference target hardware (leaderboard):** 1 x H100 80GB (not measured here)
- **Students:** r1-frieren, r1-fern, r1-tanjiro (1 logical GPU shared across all 3)

The numbers below are public reference H100 ratios from `target/program.md`. The
live RTX PRO 6000 shakedown PyTorch baseline (raw objectives) comes from the
preflight-validated baseline metrics files at
`src/eval/inference/baselines/speed/torch/<scenario>/<safe_model>/baseline_metrics.json`.
All candidate speedups must be computed against the **same** PyTorch baseline on
the same hardware.

## Current Best Per Scenario (RTX PRO 6000 shakedown)

No SENPAI launcher has been measured yet on this hardware. Starting point is the
default `vllm_running` launcher (`src/starting_points/vllm_running/start_server.sh`).
Treat that as the candidate floor; the PyTorch baseline is the speedup denominator.

| Scenario | Primary metric | Reference H100 (vLLM SMAC3) | Current best launcher (RTX PRO 6000) | W&B run | PR |
|---|---|---:|---|---|---|
| A (input-heavy, TTFT) | `scenario/A/speedup_over_pytorch` | 4.37x | unset (vLLM default) | — | — |
| B (output-heavy, TPOT) | `scenario/B/speedup_over_pytorch` | 15.23x | unset (vLLM default) | — | — |
| C (high-load, req/s) | `scenario/C/speedup_over_pytorch` | 46.70x | unset (vLLM default) | — | — |
| D (general, geomean) | `scenario/D/speedup_over_pytorch` | 5.69x | unset (vLLM default) | — | — |
| A-D aggregate | `aggregate/geomean_speedup_over_pytorch` | 11.53x | unset | — | — |

## Update History

- 2026-05-23: ledger created at advisor boot. No launchers measured yet.
