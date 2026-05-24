# InferenceBench SENPAI Live Baseline — ib-20260524-hardened-r5

Living advisor ledger. Compare every terminal review-ready PR to this state and
update when a candidate becomes the new current best for a scenario.

- **Setting**: RTX PRO 6000 Blackwell, 96GB VRAM, single GPU, 2 hour SENPAI window
- **Model**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Scoring assets (PVC)**: `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`
- **Dataset/quality seed**: 248
- **MMLU-Pro gate**: tau = 0.95 × baseline accuracy 0.298 = **observed ≥ 0.283**
- **Status of leaderboard claims**: Shakedown evidence only; not leaderboard-comparable until repeated on H100.

## PyTorch Reference (this hardware)

These are the per-scenario PyTorch baseline raw numbers measured on this GPU.
Speedup = candidate_raw / PyTorch_raw.

| Scenario | Profile | PyTorch ttft.p50 | PyTorch tpot.p50 | PyTorch req/s | Raw objective used |
|---|---|---:|---:|---:|---|
| A: input-heavy (8192/1024, burst c=1) | burst | 0.4385 s | 0.02576 s | 0.0709 | `1/ttft.p50(burst)` ≈ 2.280 |
| B: output-heavy (1024/8192, burst c=1) | burst | 0.0709 s | 0.02515 s | 0.0133 | `1/tpot.p50(burst)` ≈ 39.76 |
| C: high-load (1024/1024 ×3 profiles) | burst | 0.0702 s | 0.02368 s | 0.0845 | geomean(req/s) ≈ 0.0847 |
|   | poisson | 0.0702 s | 0.02398 s | 0.0847 |   |
|   | constant | 0.0704 s | 0.02412 s | 0.0849 |   |
| D: general (4096/2048, burst c=4) | burst | 0.2123 s | 0.02506 s | 0.0382 | geomean(1/ttft, 1/tpot, req/s) burst |

## Public H100 Reference (2026-05-21, leaderboard ladder)

| Method | Aggregate | A TTFT | B TPOT | C req/s | D geomean |
|---|---:|---:|---:|---:|---:|
| SMAC3 search 2h vLLM | 11.53× | 4.37× | 15.23× | 46.70× | 5.69× |
| TPE search 2h vLLM | 11.25× | 4.48× | 14.76× | 43.46× | 5.58× |
| vLLM default | 4.05× | 1.25× | 2.25× | 48.69× | 1.96× |
| PyTorch | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× |

These H100 ratios are context only. We score against the PyTorch raw numbers in
the row above on RTX PRO 6000.

## Current best valid launcher per scenario

No valid launcher has been measured on this branch yet. Current best per
scenario is the PyTorch baseline (1.00×) until a student posts a terminal
`SENPAI-RESULT` with a clean relaunch.

| Scenario | Current best speedup | PR | W&B run | Launcher path | Notes |
|---|---:|---|---|---|---|
| A | 1.00× | — | — | — | starting point: `src/starting_points/vllm_running/start_server.sh` |
| B | 1.00× | — | — | — | starting point: `src/starting_points/vllm_running/start_server.sh` |
| C | 1.00× | — | — | — | starting point: `src/starting_points/vllm_running/start_server.sh` |
| D | 1.00× | — | — | — | starting point: `src/starting_points/vllm_running/start_server.sh` |

## Update history

- 2026-05-24 — initialized live baseline for advisor branch `ib-20260524-hardened-r5`.
