# SENPAI Live Baseline — `ib-20260524-leasefix-r2`

- **Created:** 2026-05-24
- **Target repo:** `morganmcg1/InferenceBench-senpai` @ `codex/inferencebench-senpai-target`
- **Advisor branch:** `ib-20260524-leasefix-r2`
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Hardware (this launch):** NVIDIA RTX PRO 6000 Blackwell, ~96GB VRAM (shakedown — NOT H100 leaderboard-comparable)
- **Time budget:** 2 hours, shared 1 GPU across 3 students
- **Scoring assets:** PVC import at `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248` (preflight = pass, seed 248, 500 MMLU-Pro samples)
- **W&B project:** `wandb-applied-ai-team/inferencebench-senpai`

## Per-scenario state

| Scenario | Workload | PyTorch baseline (denominator) | Best valid launcher PR | Speedup | Quality | W&B run | Notes |
|---|---|---|---:|---:|---:|---|---|
| A | input-heavy (8K in / 1K out, burst c=1, 128 reqs) | `ttft.p50=0.4385s` | _none_ | _pending_ | _pending_ | _pending_ | open |
| B | output-heavy (1K in / 8K out, burst c=1, 64 reqs) | `tpot.p50=0.025153s` | _none_ | _pending_ | _pending_ | _pending_ | open |
| C | high-load (1K/1K, 3 profiles, 256 each) | geomean req/s ≈ 0.0847 | _none_ | _pending_ | _pending_ | _pending_ | open |
| D | balanced (4K in / 2K out, burst c=4, 96 reqs) | geomean of 1/ttft, 1/tpot, req/s | _none_ | _pending_ | _pending_ | _pending_ | open |

Starting launcher: `src/starting_points/vllm_running/start_server.sh` (vLLM defaults).
Speedup metric: `candidate_raw_objective / pytorch_baseline_raw_objective`, higher is better.

## Public H100 reference snapshot (from `program.md`, 2026-05-21)

Reference only — RTX PRO 6000 results are NOT directly comparable. Listed to indicate headroom direction.

| Method | Aggregate | Sc. A | Sc. B | Sc. C | Sc. D |
|---|---:|---:|---:|---:|---:|
| SMAC3 search 2h vLLM | 11.53x | 4.37x | 15.23x | 46.70x | 5.69x |
| Best agent (Sonnet 4.6) | 8.08x | 3.47x | 12.03x | 33.93x | 3.01x |
| vLLM default | 4.05x | 1.25x | 2.25x | 48.69x | 1.96x |

Largest H100-reference gaps from vLLM default to SMAC3: **B (2.25 → 15.23, 6.8x)**, then A (3.5x) and D (2.9x). Scenario C already wins with vLLM default — should mostly require avoiding regression.

## Update history

- 2026-05-24 — initial bootstrap; no launcher PRs yet, all scenarios open.
