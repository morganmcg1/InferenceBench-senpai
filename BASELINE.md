# SENPAI Live Baseline Ledger — InferenceBench

- **Run tag:** `ib-20260523-rerun-r4`
- **Advisor branch:** `ib-20260523-rerun-r4-advisor`
- **Target repo:** `morganmcg1/InferenceBench-senpai` (base branch `codex/inferencebench-senpai-target`)
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Benchmark GPU class:** RTX PRO 6000 Blackwell, ~96 GB VRAM (shakedown only — NOT leaderboard-comparable)
- **Budget:** 2 hour SENPAI window
- **Quality gate:** MMLU-Pro 500-question, observed/baseline ratio ≥ 0.95 (`tau = 0.95`)

> Reference snapshot (public, H100, 2026-05-21, for direction only) lives in
> `program.md`. Numbers below come from runs measured on the current hardware
> against the freshly precomputed torch PyTorch baselines on RTX PRO 6000. Do
> not compare to the H100 reference until a winner is re-measured on H100.

## Status

Scoring assets (torch speed baselines, deterministic requests, MMLU-Pro
samples, torch quality registry) are **not yet materialized** on RTX PRO 6000
for this launch. First tooling PR materializes them with
`python -m src.eval.inference.precompute_all_baselines` so subsequent
launcher PRs can compute paper-grade `speedup_over_pytorch`.

Until the baseline files exist, treat any reported speedups as raw quick-eval
viability signals only, not terminal results.

## Current best per scenario

| Scenario | Primary metric | Current best | Launcher recipe | W&B run | PR | Notes |
|---|---|---:|---|---|---|---|
| A: Input-heavy | `scenario/A/speedup_over_pytorch` | — | — | — | — | awaiting torch baseline + first launcher |
| B: Output-heavy | `scenario/B/speedup_over_pytorch` | — | — | — | — | awaiting torch baseline + first launcher |
| C: High-load | `scenario/C/speedup_over_pytorch` | — | — | — | — | awaiting torch baseline + first launcher |
| D: General | `scenario/D/speedup_over_pytorch` | — | — | — | — | awaiting torch baseline + first launcher |
| Cross-scenario | `aggregate/geomean_speedup_over_pytorch` | — | — | — | — | only after mature winners exist |

## Update history

- 2026-05-23 — Seeded BASELINE.md on advisor branch. Preflight reports missing
  torch speed baselines, request files, MMLU-Pro samples, and quality
  registry. First assignment: materialize scoring assets so paper-grade
  `speedup_over_pytorch` numbers become computable on RTX PRO 6000.
