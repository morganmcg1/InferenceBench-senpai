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
| D | balanced (4K in / 2K out, burst c=4, 96 reqs) | geomean of 1/ttft, 1/tpot, req/s = 1.9298 | [#81](https://github.com/morganmcg1/InferenceBench-senpai/pull/81) | **1.708x** | MMLU-Pro 0.292 vs 0.298 (ratio 0.980, PASS) | [42ajz9lf](https://wandb.ai/wandb-applied-ai-team/inferencebench-senpai/runs/42ajz9lf) | tanjiro, chunked-prefill 8192 + n-gram specdec 3 tok + block_size 32 + max_num_seqs 64 + FLASH_ATTN |

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

## Hardware constraints discovered this launch

- **vLLM 0.11.0 + FlashAttention v3 + FP8 KV cache is unsupported on RTX PRO 6000 (Blackwell, SM 12.0).** vLLM's `flash_attn_supports_fp8()` requires SM 9.0 (Hopper). Raises `NotImplementedError: FlashAttention does not support fp8 kv-cache on this device.` (Found by fern on PR #79.)
- **FlashInfer FP8 KV JIT-compile fails** because CCCL detects a CUDA toolkit / nvcc mismatch (`/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/include` is CUDART 12.080 but nvcc is 13.2; `CPATH` set by `senpai/runtime_env.sh` is the source). Workarounds exist (`-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK=1` or stripping the older `CPATH`) but they are off the critical path for round 1.
- **Only `VLLM_ATTENTION_BACKEND=TRITON_ATTN` supports FP8 KV cache** on this hardware/stack combination. Default to `FLASH_ATTN` for KV=auto runs; switch to `TRITON_ATTN` only when explicitly testing FP8 KV.

## Update history

- 2026-05-24 — initial bootstrap; no launcher PRs yet, all scenarios open.
- 2026-05-24 22:24 — recorded Blackwell+FA3+FP8KV constraint; reinforced fern's stop-rule for FP8 KV branch; tanjiro currently holds GPU lease for Scenario D, fern queued for B.
- 2026-05-24 23:21 — **first round-1 winner merged: PR #81 tanjiro, Scenario D, 1.708x speedup** (geomean 3.297 vs PyTorch 1.930). Quality ratio 0.980 PASS. Compounding lever for D: `--enable-chunked-prefill --max-num-batched-tokens 8192 --max-num-seqs 64 --block-size 32 --enable-prefix-caching --speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4,"prompt_lookup_min":2}'` on FLASH_ATTN. Note: pod stdout silence from 22:11→23:17 was NOT a deadlock — tanjiro's Claude session was a long-running iteration that actually did the work. Issue #90 was a false positive.
