# SENPAI InferenceBench Baseline Ledger (`ib-20260527-guard2-r1`)

- Base model: `mistralai/Mistral-7B-Instruct-v0.3`
- Time budget per run: 2h (whole research program)
- Active hardware: NVIDIA RTX PRO 6000 Blackwell-class, ~96GB VRAM (SHAKEDOWN MODE; not leaderboard-comparable)
- W&B project: `wandb-applied-ai-team/inferencebench-senpai`
- Scoring assets imported from: `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`
- Preflight (RTX PRO 6000): PASS — all scenario request files, PyTorch speed baselines, MMLU-Pro samples, and quality registry present.

## PyTorch baselines (raw)

| Scenario | Profile  | TTFT p50 (s) | TPOT p50 (s) | req/s    | gen tok/s | ok    |
|----------|----------|-------------:|-------------:|---------:|----------:|------:|
| A burst  | burst    | 0.4385       | 0.0258       | 0.0709   | 35.7      | 128/128 |
| B burst  | burst    | 0.0708       | 0.0252       | 0.0133   | 39.2      | 64/64   |
| C burst  | burst    | 0.0701       | 0.0237       | 0.0845   | 39.2      | 256/256 |
| C poisson| poisson  | 0.0697       | 0.0240       | 0.0847   | 38.4      | 256/256 |
| C constant | constant| 0.0697     | 0.0241       | 0.0849   | 38.6      | 256/256 |
| D burst  | burst    | 0.2123       | 0.0251       | 0.0382   | 38.2      | 96/96   |

## Public reference snapshot (program.md, 2026-05-21, H100, NOT directly comparable)

| Method | Aggregate | Sc. A TTFT | Sc. B TPOT | Sc. C req/s | Sc. D geomean |
|---|---:|---:|---:|---:|---:|
| SMAC3 search, 2h vLLM | 11.53x | 4.37x | 15.23x | 46.70x | 5.69x |
| vLLM default, no agent | 4.05x | 1.25x | 2.25x | 48.69x | 1.96x |

## Current best terminal results (`ib-20260527-guard2-r1`)

| Scenario | PR | Launcher | Engine | Primary metric | Speedup | Quality ratio | W&B | Notes |
|----------|----|----------|--------|----------------|--------:|--------------:|-----|-------|
| A | — | — | — | scenario/A/speedup_over_pytorch | — | — | — | open |
| B | — | — | — | scenario/B/speedup_over_pytorch | — | — | — | open |
| C | — | — | — | scenario/C/speedup_over_pytorch | — | — | — | open |
| D | — | — | — | scenario/D/speedup_over_pytorch | — | — | — | open |

## Quick / partial / failed (research signals only)

_(none yet)_

## Update history

- 2026-05-27: initialized ledger after PVC scoring-asset import + passing preflight; ready to launch first-round assignments.
