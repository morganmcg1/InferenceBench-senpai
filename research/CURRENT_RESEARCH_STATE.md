# SENPAI Research State — ib-20260524-hardened-r5

- **Date**: 2026-05-24 (RTX PRO 6000 shakedown launch, 2 hour budget)
- **Most recent human directive**: None at start; check GitHub Issues each loop.
- **Active scoring assets**: `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248` (preflight pass, exit_code=0)

## Research focus and themes

The portfolio attacks the three scenarios where vLLM defaults leave the most
headroom over PyTorch: B (output-heavy decode, ~7× gap), D (balanced, ~3× gap),
A (input-heavy prefill, ~4× gap). Scenario C is already near-saturated on the
H100 leaderboard so it is deprioritized for round 1.

Round 1 levers, picked because they have well-known multiplicative effects on
the targeted profiles and stack cleanly on a single GPU:

- **FP8 weights and FP8 KV cache** (cheap memory and bandwidth win, generally
  preserves MMLU-Pro quality when the same base model is FP8'd).
- **n-gram prompt-lookup speculative decoding** (large TPOT win on
  output-heavy decode workloads with repetitive context).
- **FlashInfer attention backend** (faster long-prefill and decode on
  Blackwell when stable).
- **CUDA graphs** (`enforce_eager=False`, default in vLLM) plus tuned
  `max_num_batched_tokens` and `max_num_seqs` for the actual concurrency.

## Open hypotheses (round 1 — 2026-05-24)

| Student | Scenario | Slug | Lever stack |
|---|---|---|---|
| r5-frieren | B | `vllm-fp8-ngram-spec-scB` | vLLM + FP8 + FP8 KV + n-gram speculative (k=5) + FlashInfer |
| r5-fern    | A | `vllm-fp8-flashinfer-prefill-scA` | vLLM + FP8 + FP8 KV + FlashInfer + tuned batched-tokens |
| r5-tanjiro | D | `vllm-fp8-spec-balanced-scD` | vLLM + FP8 + FP8 KV + n-gram speculative + balanced batching |

## Potential next research directions

These wait on round-1 evidence about which levers stack on this GPU:

- AWQ or GPTQ INT4 weight quant (often faster than FP8 on Blackwell for decode).
- EAGLE or MLP-spec decoding heads if a compatible draft is available.
- Prefix caching evaluation when sharing system prompts (mostly relevant to C).
- TensorRT-LLM and SGLang head-to-heads on the winning lever stack.
- Chunked prefill tuning sweep for A (small `max_num_batched_tokens` vs huge).
- KV cache block size and scheduler tuning for C (saturated reference, but
  RTX PRO 6000 may differ from H100).
- Re-quantize Mistral-7B-Instruct-v0.3 with calibrated FP8/INT4 if
  off-the-shelf quants regress MMLU-Pro.
