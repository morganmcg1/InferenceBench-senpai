# InferenceBench SENPAI Baseline Ledger

Live advisor-owned record of the best validated launcher per scenario on this
SENPAI launch. Compare every terminal `SENPAI-RESULT` against the entries below;
update this file when a winner becomes the new best for its scenario.

- **Launch tag:** `ib-20260523-rerun-r2`
- **Advisor branch:** `ib-20260523-rerun-r2-advisor`
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Hardware (shakedown):** 1× NVIDIA RTX PRO 6000 Blackwell, ~96 GB VRAM
- **Hardware (leaderboard):** 1× NVIDIA H100 80GB (not yet exercised in this launch)
- **Time budget per run:** 2 hours (`SENPAI_TIMEOUT_MINUTES`)
- **W&B project:** `wandb-applied-ai-team/inferencebench-senpai`
- **Quality gate:** MMLU-Pro 500q, observed ≥ 0.95 × PyTorch baseline accuracy
- **Primary metric direction:** higher is better (`speedup_over_pytorch`)

## Scenario State

| Scenario | Current best launcher | Speedup over PyTorch | PR | W&B run | Notes |
|---|---|---:|---:|---|---|
| A (input-heavy, TTFT) | — (no validated launcher yet) | — | — | — | Needs PyTorch baseline on RTX PRO 6000 |
| B (output-heavy, TPOT) | — (no validated launcher yet) | — | — | — | Needs PyTorch baseline on RTX PRO 6000 |
| C (high-load, req/s) | — (no validated launcher yet) | — | — | — | Skipped this launch — vLLM defaults already near SOTA |
| D (general, mixed) | — (no validated launcher yet) | — | — | — | Lower priority this launch |

## Public Reference Snapshot (H100, NOT comparable to RTX PRO 6000)

These are the InferenceBench public numbers from `target/program.md` (2026-05-21
snapshot, Mistral-7B-Instruct-v0.3, 2 h budget, 1× H100 80GB). They are research
direction, not a current-launch comparison. Repeat on H100 before claiming any
leaderboard-comparable result.

| Method | Aggregate | A TTFT | B TPOT | C req/s | D geomean |
|---|---:|---:|---:|---:|---:|
| SMAC3 search, 2h vLLM | 11.53× | 4.37× | 15.23× | 46.70× | 5.69× |
| TPE search, 2h vLLM | 11.25× | 4.48× | 14.76× | 43.46× | 5.58× |
| Best agent (Sonnet 4.6) | 8.08× | 3.47× | 12.03× | 33.93× | 3.01× |
| vLLM default | 4.05× | 1.25× | 2.25× | 48.69× | 1.96× |
| PyTorch baseline | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× |

Notable headline: vLLM default already wins Scenario C against search baselines
on H100. The biggest defaults-vs-search gap is Scenario B (≈6.8× headroom)
followed by Scenario D (≈2.9×) and A (≈3.5×).

## Update Protocol

When a PR posts a terminal `SENPAI-RESULT` with a clean relaunch metrics
artifact and quality gate pass:

1. Verify scenario, hardware, base model, and W&B run id.
2. If the new `speedup_over_pytorch` beats the current scenario row, replace
   the row with the new launcher path, value, PR number, and W&B run id.
3. Commit the update to `ib-20260523-rerun-r2-advisor` with a one-line message.

## Update History

- _2026-05-23 (advisor init):_ Seeded ledger. No measured RTX PRO 6000 baselines
  or candidate launchers yet for this launch.
