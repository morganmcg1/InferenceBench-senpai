# SENPAI Baseline — `ib-20260528-scen-d-r1`

Live advisor-owned baseline ledger for the Scenario D parity launch.

## Setting

- **Scenario:** D (General). LongBench-v2 prompts, 4096 input target / 2048 output target, 96 requests, burst concurrency 4, temperature 0.4.
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`.
- **GPU:** NVIDIA RTX PRO 6000 Blackwell, ~96 GB VRAM (shakedown). Not leaderboard-comparable until repeated on H100.
- **Time budget:** ~2 hours total for the launch.
- **W&B:** `wandb-applied-ai-team/inferencebench-senpai`, group `ib-20260528-scen-d-r1`.

## PyTorch reference (imported from `/mnt/new-pvc/.../rtxpro6000-seed248`)

- `generation_throughput_tokens_per_s`: 38.21
- `request_throughput_req_per_s`: 0.0382
- `ttft.p50/p90/p99`: 0.21 / 0.25 / 0.79 s
- `tpot.p50/p90/p99`: 0.025 / 0.055 / 0.127 s
- 96/96 successful, burst profile only.

## H100 reference snapshot (program.md, 2026-05-21)

| Method | Sc. D speedup |
|---|---:|
| SMAC3 vLLM 2h | 5.69x |
| TPE vLLM 2h | 5.58x |
| Random vLLM 2h | 5.42x |
| Claude Sonnet 4.6 agent | 3.01x |
| vLLM default | 1.96x |
| SGLang default | 2.14x |

This is H100, not RTX PRO 6000. Used for *direction*, not for ranking the current launch.

## Current best (live)

| Rank | PR | Engine | Recipe slug | Quick `speedup_over_pytorch` | Full `speedup_over_pytorch` | Quality ratio | W&B run | Notes |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | No terminal results yet. |

## Quick / provisional ledger

(empty)

## Failed / closed

(empty)

## Update history

- 2026-05-28: Initial scaffold created. Preflight passes for Scenario D using imported RTX PRO 6000 seed 248 assets. Two students (scen-d-frieren, scen-d-fern) idle.
