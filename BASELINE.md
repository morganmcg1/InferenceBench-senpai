# SENPAI Baseline Ledger — `ib-20260528-scen-c-r1`

Scope: **Scenario C only** (high-load, geomean request throughput across burst/poisson/constant profiles).

## Run setting

- Hardware: 1× NVIDIA RTX PRO 6000 Blackwell (~96GB VRAM) — shakedown mode, not leaderboard-comparable.
- Base model: `mistralai/Mistral-7B-Instruct-v0.3`.
- Time budget: ~2 hours from start gate (shared 1 GPU across 2 students).
- Primary metric: `scenario/C/speedup_over_pytorch` (geomean over burst/poisson/constant `request_throughput_req_per_s`).
- Quality gate: MMLU-Pro ratio ≥ 0.95 vs PyTorch baseline. Direction: speedup higher is better; quality is gate not metric.

## PyTorch baseline source (Scenario C)

- `src/eval/inference/baselines/speed/torch/inference_scenario_c_high_load/mistralai_Mistral-7B-Instruct-v0.3/baseline_metrics.json`
- Per-profile `request_throughput_req_per_s`: burst 0.0845, poisson 0.0847, constant 0.0849. Geomean ≈ 0.0848 req/s.
- MMLU-Pro torch baseline accuracy (seed=248, n=500): 0.298.

## Reference snapshot (paper, H100 80GB, 2h budget) — for orientation only

| Method | Sc. C req/s speedup |
|---|---:|
| SGLang default, no agent | 51.12x |
| vLLM default, no agent | 48.69x |
| SMAC3 search, 2h vLLM | 46.70x |
| TPE search, 2h vLLM | 43.46x |
| Random search, 2h vLLM | 41.81x |
| HF TGI default, no agent | 41.94x |
| Best agent, Claude Sonnet 4.6 | 33.93x |
| PyTorch baseline | 1.00x |

Reference points apply to H100 80GB at 2h, **not** RTX PRO 6000 shakedown. Treat as orientation, not target.

## Current best valid launcher (Scenario C)

_None yet. No terminal Scenario C results merged into this branch._

| Rank | PR | Launcher | Engine | Geomean speedup | Quality | W&B run | Notes |
|---:|---|---|---|---:|---|---|---|
| — | — | — | — | — | — | — | search just opened |

## Provisional / quick-only candidates (not yet terminal)

_None yet._

## Failed launches and dead ends

_None yet._

## Update history

- 2026-05-28: Created ledger at start of `ib-20260528-scen-c-r1`. Preflight passed against
  `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`.
