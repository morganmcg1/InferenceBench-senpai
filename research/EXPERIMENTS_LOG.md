# SENPAI Research Results — ib-20260524-ready-r5

Live log of reviewed experiments on this advisor branch. Append newest entries
at the top.

## 2026-05-24 08:48Z — PR #34: Scenario B: vLLM FP8 weights + KV cache + n-gram speculative decoding

- **Branch:** `r5-frieren/vllm-fp8-ngram-scenario-b`
- **Student:** r5-frieren
- **Hypothesis:** Stack FP8 weight quantization + FP8 KV cache + n-gram speculative decoding (5 tokens, lookup_max=4, lookup_min=2) on Scenario B (output-heavy decode, 64 reqs, 1024 in / 8192 out, concurrency 1). Expected primary lever: reduced KV cache bandwidth + speculative throughput for long decode sequences.

| Metric | This run | PyTorch baseline | Ratio |
|---|---|---|---|
| `scenario/B/speedup_over_pytorch` | **2.401×** | 1.00× | **2.40×** ↑ |
| `scenario/B/inverse_tpot_p50` (raw obj) | 95.47 | 39.76 | 2.40× |
| TPOT p50 (s) | 0.01047 | 0.02515 | 0.42× (faster) |
| TPOT p90 (s) | 0.01310 | 0.05120 | 0.26× (faster) |
| TPOT p99 (s) | 0.02952 | 0.28936 | 0.10× (10× tighter tail) |
| TTFT p50 (s) | 0.04754 | 0.07089 | 0.67× (faster) |
| Generation throughput (tok/s) | 97.33 | 39.19 | 2.48× |
| MMLU-Pro accuracy | 0.286 | 0.298 | 0.960 (≥ 0.95, **PASS**) |
| Success / total | 64 / 64 | 64 / 64 | — |

- **W&B runs:** quick `jku76a07` (2.23×), full `1pnp8qfb` (2.40×)
- **Status:** MERGED — new Scenario B best

### Analysis

FP8 weights + FP8 KV cache + n-gram speculation (5 tokens) stacked cleanly on
the decode-bound Scenario B workload. TPOT p50 dropped 2.4× and the tail tightened
10× (p99: 289 ms → 30 ms) — both expected from halved KV read bandwidth + speculative
draft acceptance. Generation throughput grew from 39 → 97 tok/s (2.5×), consistent
with the n-gram cache warming over the 64-req run.

MMLU-Pro quality margin of 96.0% (0.286/0.298) is satisfying but leaves headroom —
the uncalibrated FP8 KV `q_scale=k_scale=1.0` fallback contributes to the gap.
Computing calibrated FP8 KV scales for Mistral-7B-Instruct-v0.3 is a high-value
follow-up (could unlock more aggressive quantization + close the quality gap).

Result of **2.40×** is above the public H100 vLLM-default reference (2.25×) but
well below the SMAC3 optimized ceiling (15.23×), leaving significant headroom for:
ngram window expansion (7 tokens), EAGLE-3 or MLP-speculator replacement, FP8 KV
calibration, and chunked-prefill + larger batched-tokens for TTFT tail.

### Bugs fixed in PR

`senpai/summarize_metrics.py::_load_metrics` now unwraps `baseline.profiles` from
the scoring-assets PyTorch baseline JSON, which wraps the payload in a top-level
`"baseline": {...}` key. Without the fix, `--baseline-metrics-json` would raise
`ValueError("missing burst profile")` for every scenario.

### Suggested follow-ups (from student)

1. Increase n-gram spec to 7 tokens for Scenario B (cheap A/B)
2. Add `--enable-chunked-prefill --max-num-batched-tokens 8192` to improve TTFT tail
3. Try EAGLE-3 or MLP-speculator instead of n-gram for higher acceptance rate
4. Compute calibrated FP8 KV scaling factors for Mistral-7B-Instruct-v0.3
