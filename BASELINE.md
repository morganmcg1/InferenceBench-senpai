# SENPAI Live Baseline — ib-20260525-one3-r1

- **Research tag:** `ib-20260525-one3-r1`
- **Advisor branch:** `ib-20260525-one3-r1`
- **Target base branch:** `codex/inferencebench-senpai-target`
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Hardware (shakedown):** 1× NVIDIA RTX PRO 6000 Blackwell (~96GB VRAM)
- **Time budget:** 2 hours total (assignment → quick eval → review → full eval → cleanup)
- **W&B project:** `wandb-applied-ai-team/inferencebench-senpai`
- **Preflight:** PASS via `senpai/require_scoring_preflight.sh --import-dir /mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248 --scenario all --expected-gpu "RTX PRO 6000"` (this advisor session)

Leaderboard claims require an H100 80GB re-run. Everything below is **shakedown
evidence** until repeated on the leaderboard hardware.

## PyTorch Baselines (raw, this hardware)

These come from the pre-staged scoring assets under
`src/eval/inference/baselines/speed/torch/*` for Mistral-7B-Instruct-v0.3 on the
current RTX PRO 6000 pod.

| Scenario | Profile  | TTFT.p50 | TPOT.p50 | req/s   | gen tok/s |
|----------|----------|---------:|---------:|--------:|----------:|
| A (8K/1K, conc 1)  | burst    | 0.4385s | 0.0258s | 0.0709 | 35.65 |
| B (1K/8K, conc 1)  | burst    | 0.0709s | 0.0252s | 0.0133 | 39.19 |
| C (1K/1K, conc 64) | burst    | 0.0702s | 0.0237s | 0.0845 | 39.19 |
| C                  | poisson  | 0.0702s | 0.0240s | 0.0847 | 38.39 |
| C                  | constant | 0.0704s | 0.0241s | 0.0849 | 38.62 |
| D (4K/2K, conc 4)  | burst    | 0.2123s | 0.0251s | 0.0382 | 38.21 |

MMLU-Pro quality baseline: accuracy 0.298 (mmlu_pro, seed=248, n=500), gate
ratio threshold τ=0.95 → minimum candidate accuracy ≈ 0.283.

## Current Best Valid Launcher Per Scenario

| Scenario | Current best speedup vs PyTorch | Launcher | PR | W&B run | Notes |
|---|---:|---|---|---|---|
| A (TTFT prefill) | _none yet_ | _pending_ | _pending_ | _pending_ | Awaiting first frieren run. |
| B (TPOT decode) | _none yet_ | _pending_ | _pending_ | _pending_ | |
| C (throughput)  | _none yet_ | _pending_ | _pending_ | _pending_ | vLLM defaults are very strong here per public table. |
| D (general)     | _none yet_ | _pending_ | _pending_ | _pending_ | |

A row only moves out of "none yet" when a PR posts a terminal `SENPAI-RESULT`,
the W&B full-eval run is tied to this advisor branch, quality passes, failure
rate is low, the launcher relaunches cleanly, and the exact launcher contents
are recorded on this branch.

## Quick-Probe / Partial Evidence (not merge-eligible)

_None yet._

## Failed Launches / Closed Dead Ends

_None yet._

## Public Reference Snapshot (H100 80GB, 2026-05-21)

For orientation only. Not directly comparable to RTX PRO 6000 shakedown runs.

| Method | Aggregate | Sc. A TTFT | Sc. B TPOT | Sc. C req/s | Sc. D geomean |
|---|---:|---:|---:|---:|---:|
| SMAC3 search, 2h vLLM | 11.53x | 4.37x | 15.23x | 46.70x | 5.69x |
| TPE search, 2h vLLM | 11.25x | 4.48x | 14.76x | 43.46x | 5.58x |
| vLLM default (no agent) | 4.05x | 1.25x | 2.25x | 48.69x | 1.96x |
| PyTorch baseline | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |

## Update History

- 2026-05-25 — Advisor: initialized live baseline ledger after preflight pass.
  No measured candidates yet.
