# SENPAI Research State

- **Time:** 2026-05-25 21:06 UTC
- **Most recent human direction:** none (no GitHub Issues to advisor in this launch)
- **Research tag/branch:** ib-20260525-three2-r1
- **Hardware:** 1x RTX PRO 6000 Blackwell (~96GB VRAM, shakedown, not leaderboard-comparable to H100)
- **Time budget:** 2 hours total, ~18 min remaining (hard end 21:45 UTC; stop-new-GPU-work at 21:25 UTC)

## Round 1 final scoreboard

| Scenario | Best speedup | PR | Quality |
|---|---:|---|---|
| A: TTFT (input-heavy) | 1.28x | #108 (frieren) | quick-only, BF16 |
| B: TPOT (output-heavy) | 1.42x | #109 (fern) | quick-only, BF16 |
| C: throughput (high-load) | **21.85x** | #110 (tanjiro) | ✅ full eval, ratio 1.04 |
| D: general geomean | none | — (PR #111 closed, no activity) | — |

H100 reference ceiling: A 4.37x | B 15.23x | C 46.70x | D 5.69x (SMAC3 best, not this hardware)

## Round 1 diagnosis

The key finding from all 3 student PRs is **the same for every scenario at concurrency 1**: vLLM 0.11.0 already enables CUDA graphs and FLASH_ATTN by default on this hardware, so pure flag-tuning of the vLLM launch config provides minimal headroom vs the default. The only scenario that showed real headroom was C (high-load), where raising concurrency (max-num-seqs 256) is the orthogonal lever that vLLM default ignores.

**What can't be achieved with vLLM flag tuning alone (proven in round 1):**
- A (TTFT): chunked-prefill ON/OFF makes no difference at concurrency 1 — matmul-bound
- B (TPOT): CUDA graphs, block-size, seqs knobs all tied — kernel-bound steady-state decode
- D: unprobed, but likely similar floor given A/B findings

**Known blockers this hardware:**
- SGLang: sgl_kernel wheels missing for SM120 (RTX PRO 6000 Blackwell). Blocked until upstream precompiled binaries for SM120 are available.
- FlashInfer: disabled by runtime_env.sh for Blackwell stability.
- FP8 KV cache + FLASH_ATTN: flagged unstable combo per program.md.

## Highest-priority directions for next round

1. **Weight quantization for A and B** — AWQ or FP8-weights (not KV cache) for Mistral-7B-Instruct-v0.3. Expected 1.5-3x latency reduction on prefill (A) and decode (B). Quality gate at τ=0.95 is the risk. Frieren suggested this; fern also considered FP8 but ran out of time.

2. **Speculative decoding for B** — EAGLE3, Medusa, or n-gram draft token prediction. The biggest known TPOT lever at concurrency 1. vLLM 0.11.0 supports `--speculative-config`. Needs small draft model or n-gram approach. Fern flagged as top follow-up.

3. **Scenario D: first baseline** — vLLM with max-num-seqs 32, chunked-prefill ON, concurrency 4 as first probe. D is a general mix (4096in/2048out, c=4) that combines prefill and decode. The C winner config (high concurrency + chunked-prefill) may transfer partially to D.

4. **Scenario C continuation** — max-num-seqs sweep at 128/384/512, chunked-prefill batch-tokens sweep to push the constant-profile floor from 14.25x up. The burst/poisson profiles are already high (31.7x/23.1x); constant-profile headroom remains.

5. **SGLang unblock** — monitor upstream sgl_kernel for SM120 precompiled binaries. If they appear, SGLang's radix/lpm scheduling for C and D is a meaningful experiment.

6. **FP8 weight quantization safety validation** — before running, verify: `--quantization fp8` with FLASH_ATTN on Mistral-7B-Instruct-v0.3 on this hardware doesn't fail at import. A 1-request smoke test with quality check before committing to a full eval.

## Open questions
- Will vLLM FP8 weight quantization actually boot on SM120 without error? (Unknown — needs smoke test)
- What is the H100 gap attributable to? Clock speed diff? Memory bandwidth? Kernel availability? Important for prioritizing which experiments will have the most impact here.
