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

Sc-B torch speed baseline + MMLU-Pro torch quality registry materialized at
11:30 UTC via PR #18 (commit `4d76975`). Sc-B baseline was bounded to
`--request-limit 8` to fit the 2-hour launch window (full 64-request
sequential decode is ~4 hours wall on RTX PRO 6000). Sc-A, Sc-C, Sc-D
baselines still pending; r4-frieren is reclaiming the GPU to materialize them
after r4-fern's Sc-B vLLM eval.

Subsequent launcher PRs MUST pass `--request-limit 8` to `evaluate.py` so the
`speedup_over_pytorch` ratio is computed from sample-matched p50s.

## Current best per scenario

| Scenario | Primary metric | Current best | Launcher recipe | W&B run | PR | Notes |
|---|---|---:|---|---|---|---|
| A: Input-heavy | `scenario/A/speedup_over_pytorch` | — | — | — | — | torch baseline pending |
| B: Output-heavy | `scenario/B/speedup_over_pytorch` | torch=1.00x | torch | — | #18 | torch `tpot.p50 = 0.02875 s` (N=8), denominator for vLLM eval |
| C: High-load | `scenario/C/speedup_over_pytorch` | — | — | — | — | torch baseline pending |
| D: General | `scenario/D/speedup_over_pytorch` | — | — | — | — | torch baseline pending |
| Cross-scenario | `aggregate/geomean_speedup_over_pytorch` | — | — | — | — | only after mature winners exist |

## Torch reference numbers (RTX PRO 6000, from PR #18 @ 4d76975)

- **MMLU-Pro torch quality baseline**: `0.298` accuracy (seed=248, n=500). Quality gate `tau=0.95` → launchers must hit ≥ `0.283` observed accuracy.
- **Sc-B burst (N=8)**: ttft.p50=0.0703 s, tpot.p50=0.02875 s, itl.p50=0.01488 s, throughput=0.01246 req/s, gen_throughput=35.6 tok/s.

## Update history

- 2026-05-23 11:30 UTC — PR #18 (r4-frieren) pushed torch quality registry +
  Sc-B speed baseline at `--request-limit 8`. Preflight now PASS for
  speed_baseline_B, quality_registry, quality_samples; A/C/D still FAIL.
- 2026-05-23 09:55 UTC — Seeded BASELINE.md on advisor branch. Preflight
  reports missing torch speed baselines, request files, MMLU-Pro samples,
  and quality registry.
