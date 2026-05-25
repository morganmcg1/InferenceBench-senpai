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

## Closed (non-merging)

| PR | Scenario | Reason |
|---|---|---|
| #103 | C (high-load) | Full eval wrapper killed by `gpu_slot.py --ttl 2000` before quality phase. Speed phase finished cleanly (all 768 generations 200 OK); no quality, no metrics_full.json. Quick probe 2.82x raw — not a defensible winner. |

## Operational follow-ups for next launch

- **curand.h JIT path:** vLLM 0.11/FlashInfer first boot needs `curand.h` from
  `/usr/local/lib/python3.10/dist-packages/nvidia/curand/include/`. Symlink
  into `/usr/local/cuda/include/` or set `NVCC_PREPEND_FLAGS` in
  `senpai/runtime_env.sh` before the next launch.
- **gpu_slot.py TTL budgeting:** Full Sc C eval (256 × 3 speed profiles + 500
  MMLU-Pro) needed >2000s. Bump to 3000+ for any Sc C/D assignment.
- **Quick probe sample size:** PR #102 showed n=4 quick can overstate
  by 1.5x (5.10x → 3.50x). Default to n=16 for quick decisions.

## Active hypothesis pipeline

(window closing; no further assignments this round)

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
