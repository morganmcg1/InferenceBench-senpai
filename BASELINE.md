# SENPAI InferenceBench Baseline Ledger

Live advisor-owned baseline ledger for research tag `ib-20260523-rerun-r1` on
advisor branch `ib-20260523-rerun-r1-advisor`.

## Run Context

- **Date launched:** 2026-05-23
- **Hardware (shakedown):** 1 x NVIDIA RTX PRO 6000 Blackwell (~96 GB VRAM)
- **Model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Time budget:** 2 hours wall clock for the whole research program
- **Reference target hardware (leaderboard):** 1 x H100 80GB (not measured here)
- **Students:** r1-frieren, r1-fern, r1-tanjiro (1 logical GPU shared across all 3)

The numbers below are public reference H100 ratios from `target/program.md`. The
live RTX PRO 6000 shakedown PyTorch baseline (raw objectives) comes from the
preflight-validated baseline metrics files at
`src/eval/inference/baselines/speed/torch/<scenario>/<safe_model>/baseline_metrics.json`.
All candidate speedups must be computed against the **same** PyTorch baseline on
the same hardware.

## Current Best Per Scenario (RTX PRO 6000 shakedown)

No SENPAI launcher has been measured yet on this hardware. Round 1 produced no
leaderboard data — see `research/EXPERIMENTS_LOG.md` for the round-1 blocker
analysis. Starting point remains the default `vllm_running` launcher
(`src/starting_points/vllm_running/start_server.sh`).

The PyTorch baseline (denominator for `speedup_over_pytorch`) does NOT exist on
disk yet — round 1 attempted to build it but did not complete. Round-2 slot 1
priority is landing the torch `baseline_metrics.json` for at least one scenario
plus the MMLU-Pro quality registry.

| Scenario | Primary metric | Reference H100 (vLLM SMAC3) | Current best launcher (RTX PRO 6000) | W&B run | PR |
|---|---|---:|---|---|---|
| A (input-heavy, TTFT) | `scenario/A/speedup_over_pytorch` | 4.37x | unset (vLLM default) | — | — |
| B (output-heavy, TPOT) | `scenario/B/speedup_over_pytorch` | 15.23x | unset (vLLM default) | — | — |
| C (high-load, req/s) | `scenario/C/speedup_over_pytorch` | 46.70x | unset (vLLM default) | — | — |
| D (general, geomean) | `scenario/D/speedup_over_pytorch` | 5.69x | unset (vLLM default) | — | — |
| A-D aggregate | `aggregate/geomean_speedup_over_pytorch` | 11.53x | unset | — | — |

## Update History

- 2026-05-23: ledger created at advisor boot. No launchers measured yet.
- 2026-05-23 11:48: round 1 closed with zero leaderboard data. All 3 PRs (#20
  r1-frieren Sc C, #22 r1-fern Sc B, #23 r1-tanjiro Sc A) terminated with
  `value:null` primary metrics. See `research/EXPERIMENTS_LOG.md` for the full
  blocker analysis (vLLM 0.11→0.21 upgrade, tokenizer roundtrip patch,
  FlashInfer JIT curand.h gap). Tooling carryover (materialized request files
  for A/B/D, MMLU-Pro samples cache, `robust_truncate_messages` patch) was
  cherry-picked to advisor branch and propagates to round 2.
