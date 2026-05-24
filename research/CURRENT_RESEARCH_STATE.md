# SENPAI Research State

- **Current time:** 2026-05-24 ~17:20 (research tag `ib-20260524-hardened-r1`)
- **Most recent human research team direction:** Multiple live-run notes on PRs
  #55/#56/#57. Key constraints confirmed on RTX PRO 6000 Blackwell:
  **no FlashInfer attention** (`_sm_scale` assertion), **no FP8 KV cache**
  (`FlashAttention does not support fp8 kv-cache on this device`). Use
  `VLLM_ATTENTION_BACKEND=FLASH_ATTN`, `--kv-cache-dtype auto`, and source
  `senpai/runtime_env.sh` (sets `VLLM_USE_FLASHINFER_SAMPLER=0`,
  `VLLM_DISABLE_FLASHINFER_PREFILL=1`, `INFERENCE_BENCH_MAX_MODEL_LEN=32768`).
  FP8 weight quantization is fine.
- **Hardware setting:** RTX PRO 6000 shakedown; H100 leaderboard comparisons
  out of scope until human research team explicitly opens that mode.

## Live baseline (as of 2026-05-24 17:11)

| Scenario | Best speedup | PR | Notes |
|---|---:|---|---|
| A | **1.884x** (merged) | #57 | FLASH_ATTN + FP16 KV + FP8 weights, no chunked prefill |
| B | — | #56 WIP | r1-fern slot active until ~17:54Z |
| C | — | #55 WIP | r1-frieren waiting on slot; launcher updated to drop FP8 KV |
| D | — | — | planned for round 2 |

## Current portfolio

- **r1-tanjiro — PR #70, Sc A round 2.** Single-lever swap: TRITON_ATTN
  backend vs merged FLASH_ATTN winner. Waiting for GPU slot (~17:54Z+ after
  r1-frieren). Target: beat 1.884x.
- **r1-fern — PR #56, Sc B.** vLLM FP8 weights + FP16 KV (FP8 KV confirmed
  blocked) + n-gram spec decoding (5 tokens). GPU slot held until ~17:53Z.
  Target: `scenario/B/speedup_over_pytorch` > 1.00x.
- **r1-frieren — PR #55, Sc C.** vLLM FP8 weights + FP16 KV + max-num-seqs
  256 + chunked prefill. Updated to drop FP8 KV. Waiting for slot after
  r1-fern. Target: `scenario/C/speedup_over_pytorch` > 1.00x.

## Key hardware constraints (RTX PRO 6000, Blackwell SM 120f)

Confirmed blocked:
- `VLLM_ATTENTION_BACKEND=FLASHINFER` — runtime assertion failure in vLLM 0.11
  FlashInfer 0.6.11 (`decode_wrapper._sm_scale == self.scale`). JIT compiles
  but fails during `profile_run`. Do not retry.
- `--kv-cache-dtype fp8` — `NotImplementedError: FlashAttention does not
  support fp8 kv-cache on this device.` Do not retry with FLASH_ATTN backend.

Available and working:
- FP8 weight quantization (`--quantization fp8`): works.
- `VLLM_ATTENTION_BACKEND=FLASH_ATTN` + `--kv-cache-dtype auto`: confirmed
  working (used by merged PR #57 winner).
- `VLLM_ATTENTION_BACKEND=TRITON_ATTN`: untested, assigned to r1-tanjiro PR #70.
- n-gram speculative decoding: untested (assigned to r1-fern PR #56).
- CUDA graphs: active by default, confirmed working.

## Themes for next round (after current WIP results)

- **TRITON_ATTN viability on Blackwell.** If r1-tanjiro's PR #70 confirms it
  boots and measures faster TTFT, it opens FP8 KV with TRITON_ATTN as a future
  path (TRITON_ATTN may not have the same FP8 KV restriction as FlashAttention).
- **n-gram spec decoding on Blackwell.** r1-fern's PR #56 will confirm whether
  spec decoding (5-token n-gram) is viable for TPOT.
- **Scenario C throughput ceiling.** With FP8 KV unavailable, the concurrency
  advantage from smaller KV footprint is lost. Focus on batching policy tuning.
- **Scenario D assignment.** Once B/C have first results, r1-tanjiro can pick
  up Scenario D (balanced burst c=4) if Sc A TRITON_ATTN experiment wraps
  quickly.
- **Spec decoding on Scenario A.** With burst c=1 and no FlashInfer, n-gram
  spec on long-decode tokens after the prefill could improve TPOT.

## Potential next research directions

- TRITON_ATTN + FP8 KV for Scenario A/B once TRITON_ATTN viability confirmed.
- Scenario D balanced launcher using the validated backend (FLASH_ATTN or
  TRITON_ATTN) with burst c=4 settings.
- Cross-scenario confirmation of any A/B/C winners before claiming geomean.
- Explicit `--cuda-graph-sizes 1,2,4,8` tuning for c=1 workloads (A, B).
- `--swap-space 0` to disable CPU offload if all weights fit in 96 GB VRAM.
- SGLang fallback if vLLM hits further Blackwell-specific ceilings.
