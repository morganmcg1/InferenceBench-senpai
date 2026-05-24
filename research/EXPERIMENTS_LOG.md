# SENPAI Research Results — ib-20260524-ready-r4

## 2026-05-24 08:24 — PR #46: Scenario C vLLM FP8 KV + TRITON_ATTN (merged)

- **Branch:** r4-frieren/scenario-c-vllm-fp8-flashinfer
- **Hypothesis:** FP8 W8A8 weight quantization + FP8 KV cache + FlashInfer attention + high-concurrency continuous batching on RTX PRO 6000 Blackwell would improve Scenario C throughput over the vLLM default.
- **Results:**

| Metric | Value |
|---|---|
| scenario/C/speedup_over_pytorch | **24.40x** |
| MMLU-Pro observed | 0.288 |
| MMLU-Pro baseline | 0.298 |
| MMLU-Pro ratio | 0.966 (PASS, tau=0.95) |
| Geomean req/s | 2.0669 |
| Burst req/s | 3.187 |
| Poisson req/s | 2.201 |
| Constant req/s | 1.259 |
| Burst TTFT p50 | 93.4 ms |
| Burst TTFT p90 | 2589.8 ms (high — chunked-prefill saturation at 64 concurrency) |
| VRAM peak | 91,689 MB |
| Cold-start time | ~24 s |
| Failures | 0/768 |
| W&B run | [90r5jlvl](https://wandb.ai/wandb-applied-ai-team/inferencebench-senpai/runs/90r5jlvl) |

- **Fallbacks fired:**
  1. FlashInfer → TRITON_ATTN: FlashInfer 0.6.11 JIT cannot build kernels for sm_120 (Blackwell) on CUDA 13.2. FlashAttention v3 FP8-KV is also Hopper-only.
  2. Dropped `--quantization fp8`: FP8 W8A8 arm produced MMLU-Pro ratio 0.946 (observed 0.282 vs baseline 0.298) — just below tau 0.95. On-the-fly W8A8 quantization of Mistral-7B-Instruct-v0.3 is too lossy for greedy MMLU-Pro at this calibration level.

- **Commentary:** 24.40x is a strong result — well above the ≥20x "real winner" bar and competitive with the H100 SMAC3 reference (46.70x on different hardware). The dominant levers on Blackwell for Scenario C are (a) FP8 KV cache doubling effective cache capacity, (b) TRITON_ATTN being the only viable fp8-KV backend on this hardware, and (c) high max-num-seqs (256) + chunked-prefill ensuring the scheduler saturates the batch at all traffic patterns. The burst TTFT p90 of 2.6s suggests chunked-prefill prefill-batch saturation at c=64 — reducing max-num-batched-tokens could help. The quality loss from W8A8 on this model+tau combination is a key finding: a calibrated FP8 checkpoint would unlock the compute speedup.

- **Key operational findings:**
  - RTX PRO 6000 pod (CC 12.0, CUDA 13.2): FlashInfer JIT fails; FA3 FP8-KV Hopper-only; TRITON_ATTN is the viable backend
  - FP8 W8A8 on-the-fly fails MMLU-Pro tau=0.95 for Mistral-7B-Instruct-v0.3
  - `stdbuf -oL` required when running evaluate.py inside Claude to prevent watchdog kills (330s log-silence threshold)
  - Stale `requests_used_speed.jsonl` bug: must delete before eval rerun or eval runs only 4 requests

---

*Active: fern (PR #47 / Scenario B), tanjiro (PR #48 / Scenario A — queued)*
