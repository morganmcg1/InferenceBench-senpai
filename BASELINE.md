# SENPAI BASELINE — InferenceBench (ib-20260524-ready-r5-advisor)

This is the **live advisor-owned baseline ledger** for the
`ib-20260524-ready-r5` research tag. Update this file as soon as a new winner
becomes the current best valid launcher for a scenario.

## Run Setting

- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Hardware:** NVIDIA RTX PRO 6000 Blackwell (≈96 GB VRAM) — **shakedown only**, not leaderboard-comparable
- **Time budget:** 2 h total wall-clock for the whole research program
- **Seed:** 248 (RTX PRO 6000 scoring assets imported from `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`)
- **W&B project:** `wandb-applied-ai-team/inferencebench-senpai`
- **Research tag:** `ib-20260524-ready-r5`

## Starting launcher

- `src/starting_points/vllm_running/start_server.sh` (vLLM default)
- Flags: `--gpu-memory-utilization 0.90 --trust-remote-code --disable-log-stats`
- No FP8, no chunked prefill, no prefix caching, no speculative decoding, eager mode default off.

## PyTorch baselines (seed 248, RTX PRO 6000)

Source: `src/eval/inference/baselines/speed/torch/*/baseline_metrics.json`.
Speedup metric = `candidate_raw_objective / pytorch_baseline_raw_objective`.

| Scenario | Profile | Baseline raw objective (PyTorch) | Notes |
|---|---|---|---|
| A (TTFT input-heavy) | `burst` | `1 / ttft.p50` where `ttft.p50 ≈ 0.4385 s` → ≈ 2.28 | 128 reqs, 8192 in / 1024 out, concurrency 1 |
| B (TPOT output-heavy) | `burst` | `1 / tpot.p50` (see baseline json) | 64 reqs, 1024 in / 8192 out, concurrency 1 |
| C (req/s high load) | geomean of `burst@64`, `poisson@32 r/s`, `constant@16 r/s` | `request_throughput_req_per_s` per profile | 256 reqs per profile |
| D (general) | `burst@4` | geomean of `1/ttft.p50`, `1/tpot.p50`, `req/s` | 96 reqs, 4096 in / 2048 out, concurrency 4 |

Quality gate: MMLU-Pro 500-question subset must hit ≥ 0.95 × PyTorch baseline
accuracy (PyTorch baseline observed accuracy ≈ 0.298).

## Current best valid launcher per scenario

| Scenario | Speedup over PyTorch | Launcher | PR | W&B run | Notes |
|---|---:|---|---|---|---|
| A | (none — vLLM default ≈ 1.25× reference) | `src/starting_points/vllm_running/start_server.sh` | — | — | First wave in flight |
| B | (none — vLLM default ≈ 2.25× reference) | `src/starting_points/vllm_running/start_server.sh` | — | — | First wave in flight |
| C | (none — SGLang default ≈ 51× reference) | `src/starting_points/vllm_running/start_server.sh` | — | — | Not yet attacked |
| D | (none — vLLM default ≈ 1.96× reference) | `src/starting_points/vllm_running/start_server.sh` | — | — | First wave in flight |

Reference: the public H100 SMAC3 numbers (A 4.37×, B 15.23×, C 46.70×, D 5.69×,
aggregate 11.53×) are leaderboard-target context, not a current-best entry on
this branch.

## Update history

- 2026-05-24 — initial advisor ledger created at start of `ib-20260524-ready-r5`.
  Preflight passed against imported RTX PRO 6000 seed-248 scoring assets.
