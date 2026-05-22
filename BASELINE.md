# InferenceBench SENPAI Live Baseline (ib-20260522-r5)

- **Date:** 2026-05-22
- **Research tag:** `ib-20260522-r5`
- **Advisor branch:** `ib-20260522-r5-advisor`
- **Target setting:** Mistral-7B-Instruct-v0.3, 1× NVIDIA H100 80GB, 2 h budget per scenario, MMLU-Pro quality gate τ=0.95.
- **W&B project:** `wandb-applied-ai-team/inferencebench-senpai`

This launch starts with no SENPAI experiments completed yet. Until a terminal
SENPAI-RESULT lands, the "current best" rows below are public reference
numbers from `target/program.md`'s 2026-05-21 snapshot. Update each row as soon
as a terminal review-ready PR beats it.

## Per-Scenario Live Best

| Scenario | Primary metric | Current best speedup | Source | Notes |
|---|---|---:|---|---|
| A: Input-heavy (TTFT, 8192/1024) | `scenario/A/speedup_over_pytorch` | 4.48× | public ref: TPE search, 2h vLLM | SMAC3 4.37×, vLLM default 1.25×. Big lever: long-context prefill backend + max-num-batched-tokens. |
| B: Output-heavy (TPOT, 1024/8192) | `scenario/B/speedup_over_pytorch` | 15.23× | public ref: SMAC3 search, 2h vLLM | vLLM default 2.25×. Big lever: speculative decoding + FP8. |
| C: High-load (req/s geomean, 1024/1024) | `scenario/C/speedup_over_pytorch` | 51.12× | public ref: SGLang default | SGLang default beats every vLLM search baseline on C — search ceiling 46.70× (SMAC3). |
| D: General (geomean of 1/TTFT, 1/TPOT, req/s; 4096/2048) | `scenario/D/speedup_over_pytorch` | 5.69× | public ref: SMAC3 search, 2h vLLM | TPE 5.58×, vLLM default 1.96×. Balanced tuning. |
| Aggregate (A–D geomean) | `aggregate/geomean_speedup_over_pytorch` | 11.53× | public ref: SMAC3 search | Confirmation only, not the per-PR ranking metric. |

## Active Starting-Point Recipe

- **vLLM starting point:** `src/starting_points/vllm_running/start_server.sh`
  with `gpu_memory_utilization=0.90`, no FP8, no spec decoding, no FlashInfer,
  CUDA graphs on (eager off). Matches the leaderboard vLLM default row above.
- **SGLang starting point:** none staged yet; assign as a follow-up if a
  student wants to attack Scenario C with SGLang.
- **Speed search levers (reference only):** see
  `src/baselines/search_spaces/vllm.yaml`, `sglang.yaml`, `tgi.yaml`.

## Update History

- **2026-05-22** Bootstrap. No SENPAI runs yet; using `target/program.md`
  2026-05-21 public reference snapshot as the starting baseline.
