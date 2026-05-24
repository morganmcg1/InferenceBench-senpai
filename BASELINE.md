# InferenceBench SENPAI Baseline — ib-20260524-hardened-r1

Live advisor-owned ledger for this research tag. Updated when a terminal
review-ready PR becomes the new current best for its scenario.

- **Research tag:** `ib-20260524-hardened-r1`
- **Advisor branch:** `ib-20260524-hardened-r1`
- **Hardware setting:** RTX PRO 6000 Blackwell-class shakedown (NOT
  leaderboard-comparable; H100 confirmation required before paper claims)
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Time budget:** 2 hour InferenceBench window
- **PyTorch baseline source:** prepared scoring assets at
  `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`
  (seed 248; preflight passed 2026-05-24)

## PyTorch baseline raw objectives (RTX PRO 6000, seed 248)

| Scenario | Raw objective | Value | Notes |
|---|---|---:|---|
| A: TTFT | `1 / ttft.p50` (burst, c=1) | `2.281` /s | ttft.p50 = 0.4385 s on 8192-token prompts |
| B: TPOT | `1 / tpot.p50` (burst, c=1) | `39.76` /s | tpot.p50 = 0.0252 s on 8192-token outputs |
| C: req/s | geomean req/s across burst/poisson/constant | `~0.085` req/s | burst c=64, poisson 32, constant 16 (256 reqs each) |
| D: balanced | geomean(1/ttft, 1/tpot, req/s) burst c=4 | (composite) | ttft 0.212 s, tpot 0.0251 s, 0.0382 req/s |

`speedup_over_pytorch = candidate_raw_objective / pytorch_baseline_raw_objective`.

## Current best launcher per scenario

| Scenario | Best launcher | Speedup vs PyTorch | Quality pass | PR | W&B run | Notes |
|---|---|---:|---|---|---|---|
| A | _none yet_ | `1.00x` | n/a | — | — | first round |
| B | _none yet_ | `1.00x` | n/a | — | — | first round |
| C | _none yet_ | `1.00x` | n/a | — | — | first round |
| D | _none yet_ | `1.00x` | n/a | — | — | first round |

## Update history

- 2026-05-24 (initial) — branch bootstrapped from
  `codex/inferencebench-senpai-target` with shared-pod hardening and prepared
  RTX PRO 6000 scoring assets. No SENPAI candidates measured yet on this
  branch.
