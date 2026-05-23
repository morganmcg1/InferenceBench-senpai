# InferenceBench SENPAI — Live Baseline Ledger

Live advisor-owned baseline for `ib-20260523-rerun-r5-advisor`. Update on every
terminal `SENPAI-RESULT` that improves the current best valid launcher.

## Hardware And Model

- **Hardware:** NVIDIA RTX PRO 6000 Blackwell-class (~96 GB VRAM) — shakedown
  mode. Results are not leaderboard-comparable against the public H100 table
  until reproduced on H100.
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Time budget:** 2 hours total per scenario, supervised relaunch + `evaluate.py`
- **W&B project:** `wandb-applied-ai-team/inferencebench-senpai`
- **Research tag:** `ib-20260523-rerun-r5`

## Current State (2026-05-23)

Fresh shakedown launch. No PyTorch baselines on disk yet — `senpai/preflight.py`
fails for all four scenarios. The first tooling PR bootstraps Scenario C
PyTorch baselines and the MMLU-Pro quality baseline so that other students can
report `scenario/C/speedup_over_pytorch` from terminal `SENPAI-RESULT` markers.

## Public Reference Snapshot (H100, 2 h)

From `program.md` (2026-05-21 snapshot), Mistral-7B-Instruct-v0.3 on H100 80GB:

| Method | Aggregate | Sc. A TTFT | Sc. B TPOT | Sc. C req/s | Sc. D geomean |
|---|---:|---:|---:|---:|---:|
| SMAC3 2h vLLM | 11.53x | 4.37x | 15.23x | 46.70x | 5.69x |
| TPE 2h vLLM | 11.25x | 4.48x | 14.76x | 43.46x | 5.58x |
| Random 2h vLLM | 10.20x | 4.21x | 11.34x | 41.81x | 5.42x |
| Best agent (Claude Sonnet 4.6) | 8.08x | 3.47x | 12.03x | 33.93x | 3.01x |
| vLLM default | 4.05x | 1.25x | 2.25x | 48.69x | 1.96x |
| PyTorch baseline | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |

These are context for direction-of-effect, not leaderboard targets for
RTX PRO 6000 results.

## Current Best Valid Launcher

| Scenario | Launcher | Primary metric | W&B run | Notes |
|---|---|---:|---|---|
| A | _none_ | — | — | No PyTorch baseline yet |
| B | _none_ | — | — | No PyTorch baseline yet |
| C | _none_ | — | — | r5-frieren is bootstrapping baselines |
| D | _none_ | — | — | No PyTorch baseline yet |

## Update History

- 2026-05-23 (boot): Ledger created. No baselines on disk; first tooling PR
  bootstraps Scenario C PyTorch speed + MMLU-Pro quality assets.
