# SENPAI Research State

- **Timestamp:** 2026-05-25 (start of `ib-20260525-one4-r1` window)
- **Most recent human-team directive:** none (no GitHub Issues addressed to this tag at boot).
- **Hardware setting:** RTX PRO 6000 Blackwell shakedown (one GPU shared with frieren). Not H100 leaderboard-comparable.
- **Active students:** frieren (idle at boot).
- **Time budget:** 2-hour research program window. Reserve final 10-15 min for review/merge.

## Current research focus

Establish a strong measured baseline on each scenario and push the
highest-headroom scenarios toward the documented SMAC3/TPE 2h ceilings.

Scenario B (output-heavy) carries the largest known absolute headroom on the
public H100 snapshot (15.23x ceiling vs 2.25x vLLM default), almost entirely
from decode-time techniques: speculative decoding, CUDA graphs, KV-cache
dtype/allocation, and batch sizing. With a single benchmark GPU and one
student, opening on B is the highest-leverage first measurement.

## Merged results

| PR | Scenario | Speedup | Notes |
|---|---|---|---|
| #102 | B (output-heavy) | **3.50x** | vLLM + n-gram spec (spec=7, lookup=5), BF16 KV, CUDA graphs. Quality 1.007x. |

## Active hypothesis pipeline

1. **PR #(pending) — Scenario B / wider n-gram spec window.** Frieren. Push
   `num_speculative_tokens` to 10-12 with `prompt_lookup_max=7-8`. Calibrated
   from PR #102: quick probes on n=4 overstate; need at least n=16 quick probe.

## Potential next research directions

- **Scenario A:** vLLM long-context prefill with chunked prefill,
  prefix caching, and larger batched-token budgets. Likely 2-3x quick wins.
- **Scenario D:** mid-context balanced launcher with CUDA graphs and modest
  prefix caching.
- **Scenario C:** since defaults already exceed SMAC3, focus on stability and
  failure-rate hardening. Treat any +5% gain as a stretch goal.
- **Speculative decoding upgrades:** if n-gram looks promising on B, try EAGLE
  or Medusa draft models (if a small Mistral-compatible draft exists in
  vLLM 0.11) and self-speculative decoding through `--num-speculative-tokens`
  variants.
- **Quantization:** FP8 weights with BF16 KV (FA backend) if vLLM build
  supports it on Blackwell; defer FP8 KV cache because the FA backend
  currently rejects it.
- **Engine comparisons:** SGLang radix attention / hierarchical KV reuse for
  Scenario A long contexts; TensorRT-LLM only if time and a clean build path
  exist.
- **Throughput hardening (Scenario C):** larger `max_num_seqs`, `--no-enable-chunked-prefill`
  to verify whether chunking is hurting raw burst throughput at this concurrency.
