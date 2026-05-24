# SENPAI Research State

- **Now:** 2026-05-24, start of `ib-20260524-hardened-r4` SENPAI window
- **Latest human directive:** none yet for this tag (no open GitHub issues for the team).
- **Research focus:** LLM inference-serving optimization on InferenceBench.
  Per-scenario speedup over the PyTorch baseline (Mistral-7B-Instruct-v0.3) is
  the paper-facing metric. Quality (MMLU-Pro tau=0.95) is a hard gate.
- **Hardware context:** RTX PRO 6000 Blackwell, 96 GB VRAM, seed 248 — this is a
  shakedown launch. H100 results are required before any leaderboard claim.
  Preflight passed via PVC-imported scoring assets.
- **Pod packing:** 3 students share 1 GPU. Coordination through
  `senpai/gpu_slot.py`. Order in round 1: PR #59 (tanjiro/C) → #61 (fern/A) →
  #62 (frieren/B).

## Active round 1 hypotheses

- **#59 r4-tanjiro — Scenario C (high-load throughput).** vLLM with
  `--max-num-seqs 256`, `--max-num-batched-tokens 8192`,
  `--enable-prefix-caching`, `--kv-cache-dtype fp8`, chunked prefill.
- **#61 r4-fern — Scenario A (input-heavy / TTFT).** vLLM with large
  `--max-num-batched-tokens 16384`, chunked prefill ON, prefix caching ON,
  FlashAttention, FP8 KV.
- **#62 r4-frieren — Scenario B (output-heavy / TPOT).** vLLM with
  n-gram speculative decoding (5 spec tokens, prompt lookup 2-4) + FP8 KV +
  CUDA graphs.

## Next potential directions (round 2+)

If round-1 winners merge, push the same lever harder or compose:

- **Scenario A (prefill):** test FlashInfer backend on Blackwell, try
  `--max-num-batched-tokens 32768` to absorb the whole 8k prompt in one
  scheduling tick, try AWQ/GPTQ INT4 weights for faster prefill matmuls,
  measure how much CUDA-graph warmup is hurting first-request TTFT.
- **Scenario B (decode):** if n-gram speculative gives a real win, try EAGLE-
  or MTP-style decoding with a draft head if one is available for Mistral-7B;
  otherwise scan `num_speculative_tokens` 3/5/7 and `prompt_lookup_min`
  2 vs 3. If FP8 weights pass quality, try `--quantization fp8` for ~2x
  decode matmul speed.
- **Scenario C (throughput):** push `--max-num-seqs` to 384/512 with the 96 GB
  VRAM Blackwell card, try `--block-size 32`, try SGLang as a different
  scheduler family.
- **Scenario D (balanced):** held for cross-scenario confirmation of a mature
  winner from A/B/C, or for a balanced launcher that wins all 4.

## Backlog of bolder ideas

- TensorRT-LLM build for Mistral-7B (could unlock biggest decode speedup but
  build time may exceed remaining 2 h budget).
- Custom OpenAI-compatible server wrapping vLLM with request batching
  heuristics tuned to LongBench-v2 length distribution.
- Speculative decoding with a small draft model (e.g., a distilled Mistral 1B
  if available, otherwise n-gram is the only no-extra-checkpoint option).
- Aggressive prefix-caching tuning + admission control for Scenario C's
  poisson and constant profiles.
