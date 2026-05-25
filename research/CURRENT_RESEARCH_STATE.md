# SENPAI Research State

- **Time:** 2026-05-25 21:39 UTC
- **Most recent human direction:** none (no GitHub Issues this launch)
- **Research tag/branch:** ib-20260525-three2-r1
- **Hardware:** 1x RTX PRO 6000 Blackwell (~96GB VRAM, shakedown, not leaderboard-comparable to H100)
- **Status:** Round 2 winding down. Hard end ~21:45 UTC (~6 min). 6 PRs merged + 1 closed this launch.

## Final round 1+2 scoreboard

| Scenario | Best speedup | PR | Method | Quality validation |
|---|---:|---|---|---|
| A: TTFT (prefill) | **1.92x** | #113 ⭐ | FP8 weights + chunked-prefill | quick n=16 |
| B: TPOT (decode) | **1.46x** | #114 ⭐ | FP8 weights + decode-tight | quick n=16 |
| C: throughput | **21.85x** | #110 ✅ | BF16 vLLM tuned (256 seqs, 8192 batched tokens, chunked-prefill) | **full eval n=500** (ratio 1.04) |
| D: General | **1.41x** | #115 ⭐ | FP8 weights + balanced | quick n=16 |

H100 reference (NOT this hardware): A 4.37x | B 15.23x | C 46.70x | D 5.69x (SMAC3 best)
H100 vLLM-default: A 1.25x | B 2.25x | C 48.69x | D 1.96x

## Key learnings from round 2

**FP8 weight quantization win scaling — empirically mapped:**
- **Prefill-bound (Sc A)**: +50% (1.28x → 1.92x) — biggest win, GEMM-bound workload halved
- **Mixed (Sc D)**: +13% (1.25x → 1.41x) — proportional to prefill share at c=4
- **Decode-bound (Sc B)**: +3% (1.42x → 1.46x) — smallest win, bandwidth not GEMM is the limit
- **High-throughput (Sc C)**: inconclusive — quick-eval result (3.67x) not comparable to full-eval baseline (21.85x); needs full eval validation

**Why this ordering matters:** FP8 W8A8 dynamic on SM120 Blackwell cuts compute roughly in half via FP8 matmul kernels, but only the data already in cache is helped — for decode workloads, the bottleneck is streaming weights from HBM, which FP8 still helps (smaller load), but the benefit is bounded by memory bandwidth, not by GEMM throughput. FP8 KV cache would help decode more, but is flagged unstable + FLASH_ATTN combo on this hardware.

**FP8 boots cleanly on SM120 Blackwell** — no missing kernel issues, no fallback, full FP8 weight quantization works. This was a major uncertainty at start of round; now resolved.

**Quality gate at quick eval is unreliable** — all PRs report quality_ratio=0.839 (observed=0.250, baseline=0.298) because n=16 only has 1-2 "correct answer" granularity. This is the same noise floor for BF16 and FP8 — we cannot distinguish them at this sample size. Full n=500 eval needed before claiming FP8 introduces quality regression.

## Queued for next round (PRs #117 #118 carrying forward)

- **PR #117 (fern, Sc B)**: n-gram speculative decoding via `--speculative-config '{"method":"ngram"...}'`. The real TPOT lever for Sc B — FP8 only gave +3% because decode is bandwidth-bound, not GEMM-bound; speculative decoding directly attacks the per-token decode latency by predicting multiple tokens at once. Expected 2-3x compound on top of FP8.
- **PR #118 (tanjiro, Sc C)**: FP8 + max-num-seqs=512 (vs PR #110's 256). The FP8 weight footprint reduction frees ~7GB which can fund higher concurrency — frieren's PR #116 measured KV reporting 19.88x concurrency headroom at 32k tokens. Untapped capacity to push throughput higher than 21.85x.

## Highest-priority directions for next round (post-round-2)

1. **Full-eval validation of FP8 launchers** for A, B, D — the quick-only quality data is unreliable. A single n=500 run per scenario would tell us whether FP8 truly preserves quality (likely yes per literature, but unconfirmed on Mistral-7B-Instruct-v0.3).

2. **Full eval of PR #116's FP8 Sc C launcher** to compare to PR #110's 21.85x baseline. Frieren correctly noted the quick-eval 3.67x is structurally non-comparable to a full-eval result.

3. **Speculative decoding (PR #117 family)** — Sc B's bandwidth-bound bottleneck is best attacked by predicting multiple decode tokens per forward pass. N-gram is the cheapest path; EAGLE3 and Medusa are higher-effort follow-ups.

4. **Sc C concurrency stacking (PR #118 family)** — push max-num-seqs to 384/512 to use the freed KV pool.

5. **Alternative weight quantizations** — AWQ Mistral-7B-Instruct-v0.3 vs FP8 W8A8 vs int8_w8a8. AWQ pre-quantized weights typically preserve quality better than runtime FP8 dynamic, may pass the n=500 quality gate cleanly.

6. **SGLang unblock** — monitor upstream sgl_kernel for SM120 precompiled binaries. SGLang's radix prefix tree and lpm scheduling are different lever sets that could compound with FP8.

7. **Cross-scenario confirmation** — once a launcher is fully validated in one scenario, run a single A→D pass to see if recipe transfers.

## Open questions for next round

- Does FP8 weight quantization actually preserve quality at full n=500 MMLU-Pro? (Critical for paper-facing baseline)
- Does FP8 Sc C full-eval beat 21.85x baseline? (Most important throughput question)
- Does n-gram speculative decoding actually boot and improve Sc B? (Highest expected upside for the smallest-win scenario)
- Can we get a full-eval terminal for A, B, D in next round? (Round 2 was all quick-eval terminals — quick-eval ratios are not paper-grade)
