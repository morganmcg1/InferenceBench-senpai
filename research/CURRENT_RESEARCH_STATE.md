# SENPAI Research State

- **Current time:** 2026-05-24 (research tag `ib-20260524-hardened-r1`)
- **Most recent human research team direction:** none yet on this branch.
  Defaults to the InferenceBench target contract in `target/program.md`.
- **Hardware setting:** RTX PRO 6000 shakedown; H100 leaderboard comparisons
  out of scope until human research team explicitly opens that mode.

## Current research focus

This is the first SENPAI round on a fresh `ib-20260524-hardened-r1` branch.
Goal: drive each scenario's `speedup_over_pytorch` above 1.0x and start a real
ledger in `BASELINE.md` against the PyTorch baseline metrics shipped in the
prepared scoring assets.

Three idle students, one shared GPU, 2 hour total wall clock. Coordinate heavy
runs through `senpai/gpu_slot.py`; serial heavy workloads only.

## Round 1 portfolio (one scenario per student)

- **r1-frieren — Scenario C (high-load throughput).** vLLM FP8 weights +
  FP8 KV cache + high `max-num-seqs`, prefix caching, chunked prefill. Largest
  expected headroom relative to PyTorch (reference snapshot shows >40x req/s
  for tuned vLLM at the H100 setting).
- **r1-fern — Scenario B (output-heavy TPOT).** vLLM FP8 + FP8 KV + n-gram
  speculative decoding for long-decode latency on burst c=1. Reference
  snapshot shows >14x for tuned search on H100.
- **r1-tanjiro — Scenario A (input-heavy TTFT).** vLLM FP8 + FP8 KV +
  FlashInfer attention backend + no chunked prefill + large
  `max-num-batched-tokens`. Reference snapshot shows ~4x for tuned search on
  H100.

Scenario D held for round 2 once we know which engine settings are stable on
this hardware.

## Themes worth attacking next (after round 1 results)

- **Quantization quality vs speed tradeoff.** Confirm FP8 weights+KV passes
  the 0.95 MMLU-Pro gate; if not, fall back to FP16 KV with FP8 weights.
- **Attention backend by scenario.** FlashInfer for prefill-heavy A, FlashAttn
  for decode-heavy B, possibly TRITON_ATTN as a comparison.
- **Speculative decoding shape.** n-gram tokens (3, 5, 7) and draft length on
  B; light spec on D burst c=4; whether spec helps under concurrency on C.
- **Batching vs latency tradeoff for C.** max-num-seqs 128/256/512 and
  max-num-batched-tokens 8192/16384, possibly chunked prefill off for
  throughput at the cost of TTFT.
- **CUDA graphs and enforce-eager interaction with FP8 quantization on
  Blackwell.** Worth confirming kernels compile cleanly on RTX PRO 6000.

## Potential next research directions

- Scenario D balanced launcher once A/B/C have validated FP8 + KV cache
  defaults.
- Cross-scenario confirmation runs on any A-D winners before claiming a
  geomean number.
- SGLang and TGI sanity checks if vLLM hits unexpected ceilings on this
  hardware (vLLM is the priority because the search spaces and reference
  snapshot favor it on the leaderboard).
- Prefix-caching ablations on C if requests show strong template overlap.
- Targeted CUDA graph warmup / capture-size tuning if first round shows long
  warmup or first-request tail latency.
