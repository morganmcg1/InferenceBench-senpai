# SENPAI BASELINE — ib-20260528-scen-a-r1

Live advisor-owned baseline ledger. Updated only from full clean-relaunch
results validated by `senpai/validate_result.py`.

## Run context

- **Research tag:** `ib-20260528-scen-a-r1`
- **Advisor branch:** `ib-20260528-scen-a-r1`
- **Hardware:** NVIDIA RTX PRO 6000 (~96 GB VRAM, shakedown — not H100 leaderboard-comparable)
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Scenario in scope:** A only (input-heavy, long-context prefill / TTFT)
- **Time budget:** 2h, gate opened `2026-05-28T16:36:57Z`, cutoff `2026-05-28T18:36:57Z`
- **Seed / scoring assets:** seed 248, imported from PVC
- **PyTorch speed baseline:** `src/eval/inference/baselines/speed/torch/inference_scenario_a_input_heavy/mistralai_Mistral-7B-Instruct-v0.3/baseline_metrics.json`
- **PyTorch quality baseline:** `src/eval/inference/baselines/quality/mistralai_Mistral-7B-Instruct-v0.3_torch.json` (MMLU-Pro accuracy ≈ 0.298, τ=0.95)

## PyTorch raw objective (for speedup conversion)

| Scenario | Profile | Raw metric | Value | Source |
|---|---|---|---:|---|
| A | burst | `ttft.p50` (s) | 0.4385 | baseline_metrics.json |
| A | burst | `1 / ttft.p50` (req/s, higher better) | 2.281 | derived |

## Current best valid launcher (full eval, validated)

| Scenario | Speedup over PyTorch | TTFT.p50 (s) | MMLU-Pro ratio | PR | W&B run | Launcher |
|---|---:|---:|---:|---|---|---|
| A | _none yet_ | — | — | — | — | — |

## Reference snapshot (H100, paper-public)

| Method | Sc. A TTFT speedup |
|---|---:|
| SMAC3 search, 2h vLLM | 4.37x |
| TPE search, 2h vLLM | 4.48x |
| Best listed agent (Claude Sonnet 4.6) | 3.47x |
| vLLM default, no agent | 1.25x |
| PyTorch baseline | 1.00x |

These are paper-public H100 numbers and are not directly comparable to current
RTX PRO 6000 shakedown results. They indicate where the meaningful headroom is.

## Update history

- 2026-05-28T16:40Z — initialized BASELINE.md from preflight import (no current best yet).
