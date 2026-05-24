# SENPAI Research Results

## 2026-05-24 07:58 — PR #43: Scenario B n-gram speculative decoding + FP8 KV + CUDA graphs

- **Branch:** r2-frieren/B-ngram-spec-fp8-kv
- **Hypothesis:** Scenario B (output-heavy, 1024→8192, concurrency 1) is TPOT-bound. n-gram speculative decoding (prompt_lookup, no draft model) + FP8 KV cache + CUDA graphs should deliver a strong TPOT win.
- **Result:**

| Metric | Observed | PyTorch Baseline | Ratio |
|---|---:|---:|---:|
| `scenario/B/speedup_over_pytorch` | **2.913x** | 1.00x | +191% |
| `tpot.p50` (s/token) | 0.00864 | 0.02515 | 2.91x faster |
| `tpot.p90` | 0.02293 | 0.05120 | 2.23x |
| `ttft.p50` (s) | 0.0648 | 0.0709 | 1.09x |
| `ttft.p90` | 0.2028 | 0.1476 | 0.73x (slower) |
| `generation_throughput` (tok/s) | 98.28 | 39.19 | 2.51x |
| `request_throughput` (req/s) | 0.0237 | 0.0133 | 1.78x |
| `quality/mmlu_pro_observed_accuracy` | **0.3203** | 0.298 | 1.075 — PASS |
| `vram_peak_mb` | 89,593 | — | — |
| `failure_count` | 0/8 | 0/64 | — |

- **W&B run:** `fgzox7fi` (group: B-ngram-spec, state: finished)
- **Eval mode:** `--quick`, n=8 burst requests (not scenario-default n=64; see notes)
- **Status:** MERGED into advisor branch (squash-merge, commit 7b73b88)

### Analysis & Conclusions

**Result beats H100 vLLM-default reference (2.25x)** and passed the quality gate with margin (ratio 1.075). Missed the 4x target.

**What worked:**
- n-gram speculative decoding via `{"method":"ngram","num_speculative_tokens":5}` contributed meaningful TPOT acceleration (~2x at p50 before CUDA graph overhead)
- CUDA graphs captured cleanly; cold relaunch ~17s (graph-cached), contributing an estimated 1.3–1.5x factor of the 2.91x total
- MMLU-Pro quality fully preserved — speculative decoding is mathematically lossless on the greedy/low-temp distribution

**What fell short:**
- **FP8 KV cache was inactive** — CUDA 13 nvcc on the pod is incompatible with FlashInfer 0.6.11's bundled CUDA 12 headers. The `IB_KV_CACHE_DTYPE=fp8` env-var path in the launcher is wired but the FlashInfer JIT fails, forcing fallback to bf16 KV. VRAM stayed near ceiling (89.6/96 GB) rather than freeing ~10–15 GB.
- **FlashInfer attention backend also broken** for same reason. FLASH_ATTN fallback used throughout.
- **prompt_lookup acceptance rate is variance-heavy**: 4 "easy" prompts at smoke time gave 3.70x; 8 prompts gave 2.91x. The expected value at n=64 scenario-default is likely 2.5–3.2x, not 4x. prompt_lookup is the weakest spec family; a real draft model (EAGLE/Medusa) is needed to reach the SMAC3 15x regime.
- **n=8 quick eval** — The scenario's 8192-token output at 98 tok/s would take ~89 min for n=64 requests. Quick eval (n=8) is the only realistic option within the 35-min slot. Metric is statistically reasonable but has higher variance than a full run.

**Key hardware note:** CUDA 13 nvcc / FlashInfer CUDA-12 header mismatch affects ALL students on this pod. FP8 KV and FLASHINFER backend are both blocked for this session unless nvcc is upgraded or FlashInfer is patched.

### Round-2 follow-ups for Scenario B

1. **num_speculative_tokens=3** — Smaller lookahead trades acceptance breadth for less wasted verifier compute; expected +5–10% TPOT at more diverse prompts.
2. **EAGLE draft-model PR** — Mistral-7B EAGLE weights exist on Hub; this is the path to 10x+ TPOT regime.
3. **Full n=64 eval** in a longer slot to confirm the n=8 estimate and reduce variance.
4. **Increase prompt_lookup_max** from 4 to 6–8 for longer lookups on high-repetition outputs.
