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
None yet. Awaiting terminal full eval from PR #159.

## Quick / provisional ledger
Quick evals (n=16 screening, quality skipped — NOT terminal):

| Source | Arm | Raw TPOT p50 (s) | Speedup vs PT | MMLU-Pro (screening) | W&B |
|---|---|---:|---:|---|---|
| PR #159 frieren | F0 vLLM default | ~0.0175 | 1.44x | 0.25 / 0.298 = 0.84 (n=16) | o39u89pa |
| PR #159 frieren | F1 n-gram k=5 | ~0.0072 | **3.50x** | 0.25 / 0.298 = 0.84 (n=16) | 3rvxd2tx |
| PR #159 frieren | F2 n-gram k=3 | ~0.0077 | 3.28x | 0.19 / 0.298 = 0.63 (n=16) | smaj0ntn |
| PR #162 fern   | — | — | — | none yet | — |

F1 is the strongest quick candidate; promoted to full eval. Quality
screening uses n=16 and is noise-floor adjacent (F0 vLLM default also at
0.84) — full eval (n=500) will be the real gate.

## Failed launches
None confirmed yet. F2 quality screening at 0.63 (n=16) is borderline — would need full eval to confirm whether n-gram k=3 actually degrades quality or it was small-sample noise.

## Update history
- 2026-05-28 16:40 UTC — Branch opened; preflight passes via
  `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`;
  PyTorch baseline metrics available; quality registry pinned.
- 2026-05-28 17:25 UTC — Frieren posted 3 quick evals (F0/F1/F2); F1
  n-gram k=5 leading at 3.50x raw speedup. Fern silent — advisor comment
  posted requesting status.
