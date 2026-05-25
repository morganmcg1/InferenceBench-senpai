# SENPAI Research Results

## 2026-05-25 08:29 — PR #100: Scenario A: vLLM long-context prefill tune

- **Branch:** frieren/a-vllm-prefill-tune
- **Student:** frieren
- **Hypothesis:** Tune vLLM for Scenario A (8K prefill / 1K decode, concurrency 1,
  burst TTFT.p50). Key levers: FlashAttention backend, `--max-num-batched-tokens 16384`
  to avoid chunked-prefill overhead, CUDA graphs ON, tight `--max-model-len 10240`,
  no prefix caching.

### Results

| Arm | Variant | TTFT.p50 | Speedup vs PyTorch | Quality | W&B |
|---|---|---:|---:|---|---|
| 1 (terminal, full) | chunked prefill ON, batched_tokens=16384 | 0.3527s | **1.2433x** | PASS (ratio 1.00) | `x6t65ria` |
| 2 (quick only) | monolithic prefill (--no-enable-chunked-prefill) | ~0.345s | ~1.271x | PASS | n/a |

Full eval detail (Arm 1, W&B run `x6t65ria`):

| Metric | Arm 1 | PyTorch Baseline |
|---|---:|---:|
| TTFT.p50 | 0.3527s | 0.4385s |
| TPOT.p50 | 0.0172s | 0.0258s |
| req/s | 0.0908 | 0.0709 |
| gen tok/s | 55.40 | 35.65 |
| success / fail | 128 / 0 | 128 / 0 |
| VRAM peak | 90.2GiB | — |
| MMLU-Pro accuracy | 0.298 (ratio 1.00) | 0.298 |

### Analysis and Conclusions

**Merged as first Scenario A winner (1.2433x).**

1. **Chunked vs monolithic prefill neutral at concurrency 1.** Arms 1 and 2 tied
   at 1.271–1.272x in quick probes. This confirms the theoretical expectation:
   chunked prefill gains come from interleaving prefill with decode for concurrent
   requests — at concurrency 1 there is nothing to interleave with.

2. **Speed gain source.** The ~24% improvement over PyTorch likely comes almost
   entirely from vLLM's FlashAttention kernel vs PyTorch's naive eager attention.
   The explicit tuning flags (large batched-tokens, tight max_model_len) prevent
   _regressions_ relative to naive vLLM defaults but don't unlock large additional
   gains beyond FlashAttention.

3. **TPOT bonus (1.50x).** TPOT.p50 improved from 0.0258s to 0.0172s — a 50%
   decode-speed win for free. This suggests vLLM CUDA graphs and scheduling are
   efficient for the 1K-token decode segment of Scenario A.

4. **VRAM ceiling.** 90.2GiB with `--gpu-memory-utilization 0.90` on ~98GiB
   RTX PRO 6000. Headroom is ~8GiB. An H100 80GB leaderboard run would need
   `--gpu-memory-utilization ~0.75` to stay under budget.

5. **Gap to HPO target.** Public SMAC3 4.37x on H100 vs our 1.243x on RTX PRO 6000.
   Most of that gap is likely attention kernel differences (FlashInfer vs FlashAttn,
   FP8 weights, hardware arch). The remaining search surface for Scenario A:
   - Attention kernel: Triton (`TRITON_ATTN`) or FlashInfer (currently disabled on
     RTX PRO 6000 by runtime_env.sh for stability)
   - FP8 weight quantization (needs quality validation)
   - Speculative decoding with n-gram lookup (less helpful for A's short decode,
     but the 1K decode might benefit from even 3–5 speculative tokens)
   - Parallelism / disaggregation: only 1 GPU so not applicable here

6. **Next priority: Scenario B.** TPOT optimization (8K decode, concurrency 1) is
   the highest-headroom scenario (default 2.25x → HPO 15.23x on H100). The free
   TPOT improvement seen in Scenario A confirms the decode path is responsive to
   tuning. N-gram speculative decoding is the primary lever to try.
