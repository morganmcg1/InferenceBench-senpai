# SENPAI InferenceBench Baseline Ledger

- **Advisor branch:** `ib-20260528-12h-r2`
- **Research tag:** `ib-20260528-12h-r2`
- **Hardware (current):** NVIDIA RTX PRO 6000 (~96 GB VRAM) — shakedown only,
  not leaderboard-comparable to H100 reference snapshot.
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **W&B:** `wandb-applied-ai-team/inferencebench-senpai`
- **Scoring assets:** imported from
  `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`
  (seed 248, all four scenarios). Preflight: PASS as of 2026-05-28 10:42 UTC.

Reference snapshot for the H100 80GB / 2-hour-per-run leaderboard setting
(2026-05-21, copied from `program.md` for orientation only — not RTX PRO 6000
baselines):

| Method | Sc. A TTFT | Sc. B TPOT | Sc. C req/s | Sc. D geomean |
|---|---:|---:|---:|---:|
| PyTorch baseline | 1.00x | 1.00x | 1.00x | 1.00x |
| vLLM default | 1.25x | 2.25x | 48.69x | 1.96x |
| SGLang default | 1.22x | 1.77x | 51.12x | 2.14x |
| HF TGI default | 1.14x | 1.37x | 41.94x | 1.80x |
| Random search vLLM 2h | 4.21x | 11.34x | 41.81x | 5.42x |
| TPE search vLLM 2h | 4.48x | 14.76x | 43.46x | 5.58x |
| SMAC3 search vLLM 2h | 4.37x | 15.23x | 46.70x | 5.69x |
| Sonnet 4.6 agent | 3.47x | 12.03x | 33.93x | 3.01x |

## Current best valid launcher (RTX PRO 6000 shakedown)

_No advisor-validated terminal results yet for this branch. PyTorch baseline is
the only currently-known floor on this hardware._

| Scenario | Primary metric | Best speedup vs PyTorch | Launcher | W&B run | PR |
|---|---|---:|---|---|---|
| A | scenario/A/speedup_over_pytorch | 1.00x (PyTorch floor) | — | — | — |
| B | scenario/B/speedup_over_pytorch | 1.00x (PyTorch floor) | — | — | — |
| C | scenario/C/speedup_over_pytorch | 1.00x (PyTorch floor) | — | — | — |
| D | scenario/D/speedup_over_pytorch | 1.00x (PyTorch floor) | — | — | — |

PyTorch baseline raw objectives (from imported metrics):

| Scenario | Profile | TTFT.p50 | TPOT.p50 | Throughput (req/s) |
|---|---|---:|---:|---:|
| A | burst | 0.4385s | 0.0258s | 0.0709 |
| B | burst | 0.0709s | 0.0252s | 0.0133 |
| C | burst | 0.0702s | 0.0237s | 0.0845 |
| C | poisson | 0.0702s | 0.0240s | 0.0847 |
| C | constant | 0.0704s | 0.0241s | 0.0849 |
| D | burst | 0.2123s | 0.0251s | 0.0382 |

## Provisional / quick-only results

_(none yet)_

## Update history

- 2026-05-28 10:42 UTC — Created the baseline ledger. Preflight passed for all
  scenarios; assigning first round of experiments to fern, frieren, tanjiro.
