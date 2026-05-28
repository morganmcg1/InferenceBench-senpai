# SENPAI BASELINE — Scenario B (Output-Heavy)

- Branch: `ib-20260528-scen-b-r1`
- Launched: 2026-05-28
- Setting: RTX PRO 6000 Blackwell shakedown (NOT leaderboard-comparable to H100)
- Base model: `mistralai/Mistral-7B-Instruct-v0.3`
- Scenario: B (input ~1024 / output ~8192, 64 requests, burst, concurrency 1)
- Primary metric: `scenario/B/speedup_over_pytorch` = `(1/candidate_tpot_p50_burst) / (1/baseline_tpot_p50_burst)`
- Quality gate: MMLU-Pro 500-question subset, tau=0.95 of PyTorch baseline accuracy 0.298

## PyTorch reference baseline (must beat this to score speedup>1)
- File: `src/eval/inference/baselines/speed/torch/inference_scenario_b_output_heavy/mistralai_Mistral-7B-Instruct-v0.3/baseline_metrics.json`
- `requests_sha256`: `6a7bb79e0cde7cf24334c6b60da319c7bf458c756e75067f28c1770e0f4b869d`
- Burst TPOT p50 = **0.02515 s/token** → raw objective `1/tpot.p50` = **39.76 tok/s**
- TTFT p50 = 0.0709 s; ITL p50 = 0.0146 s; gen throughput = 39.19 tok/s
- 64/64 successful requests, failure rate 0.0

## Public reference (2026-05-21 H100, 2h budget) — NOT directly comparable to RTX PRO 6000
| Method | Sc. B TPOT speedup |
|---|---:|
| SMAC3 search, 2h vLLM | 15.23x |
| TPE search, 2h vLLM | 14.76x |
| Random search, 2h vLLM | 11.34x |
| Best listed agent (Sonnet 4.6) | 12.03x |
| vLLM default, no agent | 2.25x |
| SGLang default, no agent | 1.77x |
| HF TGI default, no agent | 1.37x |
| PyTorch baseline | 1.00x |

Reference H100 numbers above are search direction only. RTX PRO 6000 Blackwell
results can differ; need on-hardware measurement.

## Current best terminal launcher (this branch)
None yet. First experiments are launching.

## Quick / provisional ledger
None yet.

## Failed launches
None yet.

## Update history
- 2026-05-28 16:40 UTC — Branch opened; preflight passes via
  `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`;
  PyTorch baseline metrics available; quality registry pinned.
