# SENPAI InferenceBench Baseline Ledger

Advisor branch: `ib-20260524-ready-r3-advisor`
Research tag: `ib-20260524-ready-r3`
Last updated: 2026-05-24

## Active Setting

- **Mode:** RTX PRO 6000 shakedown (not leaderboard-comparable)
- **GPU:** NVIDIA RTX PRO 6000 Blackwell, ~96 GB VRAM
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Time budget:** 2 hours total research clock
- **Scoring assets seed:** `rtxpro6000-seed248` (preflight imported and passed)
- **Starting launcher:** `src/starting_points/vllm_running/start_server.sh` (vLLM
  default with `--gpu-memory-utilization 0.90 --trust-remote-code
  --disable-log-stats`, `--max-model-len 131072`)

## PyTorch Baseline (raw objective sources)

Located at `src/eval/inference/baselines/speed/torch/inference_scenario_*/mistralai_Mistral-7B-Instruct-v0.3/baseline_metrics.json`.
Quality baseline MMLU-Pro accuracy = 0.298 (`src/eval/inference/baselines/quality/mistralai_Mistral-7B-Instruct-v0.3_torch.json`),
quality gate ratio τ = 0.95.

## Current Best Valid Launchers

No SENPAI winners merged yet — no PRs reviewed on this advisor branch.

| Scenario | Primary metric (paper-facing) | Current best speedup vs PyTorch | Source PR | W&B run | Notes |
|---|---|---:|---|---|---|
| A: Input-heavy | `scenario/A/speedup_over_pytorch` (`1/ttft.p50` burst) | — (use vLLM default as starting reference; not yet measured this run) | starting point | — | Headroom vs reference snapshot: large (vLLM default ~1.25x, SMAC3 ~4.48x on H100) |
| B: Output-heavy | `scenario/B/speedup_over_pytorch` (`1/tpot.p50` burst) | — | starting point | — | Largest headroom (vLLM ~2.25x, SMAC3 ~15.23x on H100) |
| C: High-load | `scenario/C/speedup_over_pytorch` (geomean req/s across burst/poisson/constant) | — | starting point | — | vLLM default already very strong on reference (~48.7x) — small headroom |
| D: General | `scenario/D/speedup_over_pytorch` (geomean of 1/ttft.p50, 1/tpot.p50, req/s burst) | — | starting point | — | Medium headroom (vLLM ~1.96x, SMAC3 ~5.69x on H100) |
| Aggregate | `aggregate/geomean_speedup_over_pytorch` | — | — | — | Only run on mature cross-scenario winners |

## Update History

- 2026-05-24 — Ledger created on advisor boot. RTX PRO 6000 preflight passed
  via `--import-dir /mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`.
  All 3 students idle; first assignment round opens with scenario-targeted launcher hypotheses.
