# SENPAI Research Results — `ib-20260525-one4-r1`

Per-PR experiment ledger. Each entry records the hypothesis, results, and
analysis once the PR returns terminal `SENPAI-RESULT` evidence.

<!-- Entries will be appended chronologically below. -->

## 2026-05-25 13:57 — PR #102: Sc B: vLLM + n-gram spec decoding + CUDA-graphed BF16 KV launcher

- **Branch:** `frieren/sc-b-vllm-ngram-spec-bf16`
- **Hypothesis:** Scenario B (output-heavy, 1024 in / 8192 out, burst c=1) is decode-bound. vLLM with CUDA graphs + BF16 KV + n-gram speculative decoding (spec_tokens=7, prompt_lookup 2–5) should substantially beat the PyTorch baseline TPOT.
- **W&B run:** `x9t8u1d6` (group `scenario-B-ngram-spec-decoding`)

### Results

| Metric | Value | PyTorch baseline | Change |
|---|---|---|---|
| `scenario/B/speedup_over_pytorch` | **3.497x** | 1.00x | +3.50x |
| `burst.tpot.p50` | 0.00719 s | 0.02515 s | **3.50x faster** |
| `burst.tpot.p90` | 0.01221 s | — | — |
| `burst.generation_throughput` | 129.26 tok/s | 39.19 tok/s | +3.30x |
| `burst.itl.p50` | 0.01296 s | 0.01462 s | 1.13x |
| `success / total` | 64/64 | 64/64 | = |
| `failure_rate` | 0 | 0 | = |
| `mmlu_pro_observed` | 0.300 | 0.298 | ratio 1.007 ✅ |
| VRAM peak | 89,759 MB | — | — |

### Quick-probe summary

| Arm | spec_tokens | lookup_max | quick tpot.p50 | quick speedup |
|---|---|---|---|---|
| 1 | 5 | 4 | 0.00907 s | 2.77x |
| 2 (promoted) | 7 | 5 | 0.00492 s | 5.10x |

Quick arm 2 showed 5.10x but full eval landed at 3.50x — n-gram acceptance rate dilutes from 4 requests to 64. Important calibration: quick probes on n=4 overstate n-gram benefit.

### Analysis

Good first result. vLLM CUDA graphs + n-gram spec decoding clears the 3.5x mark on this Blackwell pod with zero failures and quality well within gate. The gap to the H100 SMAC3 15.23x reference is real and reachable via:
1. Better spec decoding (EAGLE-class draft heads)
2. Wider n-gram lookup window (spec_tokens=10-12, lookup_max=8)
3. FP8 weight quantization (with FA backend; KV remains BF16)
4. Larger `max_num_batched_tokens` for prefix amortization

ITL improvement (1.13x) vs TPOT improvement (3.50x) is expected: spec decoding reduces per-step count (TPOT wins) but the gap between accepted chunks still has wall-clock overhead (ITL lags).

### Verdict: MERGED
