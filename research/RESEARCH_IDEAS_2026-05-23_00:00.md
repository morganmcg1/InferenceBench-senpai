# Research Ideas — ib-20260523-rerun-r5 boot

Generated at advisor boot for the 2026-05-23 rerun-r5 launch. The active round
focuses Scenario C only, but these ideas seed follow-up rounds once baselines
exist.

## Round 1 (in flight, see CURRENT_RESEARCH_STATE.md)

Scenario C × 3 students:

1. r5-frieren — torch baseline + MMLU-Pro quality bootstrap (tooling).
2. r5-fern — vLLM throughput launcher, FP16 KV, max-num-seqs=256.
3. r5-tanjiro — vLLM launcher, FP8 KV cache, max-num-seqs=512, FlashInfer.

## Round 2 candidates (depend on r5-frieren torch baseline existing)

### Scenario C follow-ups (highest leverage)

C-a. **Larger batch / higher concurrency** — `--max-num-seqs 768`, `mnb-tokens
24576`, `gpu-util 0.95`. Only if r5-tanjiro's 512+fp8 passes quality.

C-b. **Speculative decoding** — n-gram speculator with
`--speculative-config '{"method":"ngram","num_speculative_tokens":5,"ngram_prompt_lookup_max":4}'`
on top of the best non-speculative recipe. Speculative is the single biggest
known throughput lever on Mistral-7B-class decode-bound traffic; only adds
overhead under low-concurrency settings.

C-c. **Block size 32 vs 16** — `--block-size 32` on the winning recipe.
Larger blocks reduce PagedAttention bookkeeping at the cost of internal
fragmentation; Scenario C decodes 1024 tokens, so paging overhead is
amortized over many block fills.

C-d. **Attention backend ablation** — Triton vs FlashInfer vs FlashAttention
on the winning recipe, holding everything else fixed. Cheap to run and gives
us a clean cross-backend baseline for future picks.

C-e. **Chunked prefill off** — Test whether `--no-enable-chunked-prefill`
improves throughput-only workloads (prefix caching off, no shared prompts).
Chunked prefill was designed for mixed prefill/decode; if it interrupts
decode under burst concurrency it could hurt p50 throughput.

### Scenario A (needs A torch baseline first)

A-a. **Long-context prefill optimization** — bootstrap A torch baseline (128
requests × 8K input), then launch vLLM with `--max-num-batched-tokens 32768`
and FlashAttention. TTFT-dominant scenario.

A-b. **Speculative decoding for prefill** — vLLM 0.10+ supports speculative
prefill via medusa or similar; only one prompt at a time so the gain is from
reducing serial prefill latency rather than batched throughput.

### Scenario B (needs B torch baseline first)

B-a. **Decode-optimized launcher** — `max-num-seqs 64-128` (concurrency 1
in B), CUDA graphs critical, fp8 KV cache, possibly fp8 model weights if
available for Mistral-7B-Instruct-v0.3.

B-b. **Speculative decoding (medusa/eagle)** — B is the textbook speculative
decoding scenario: long decodes, low concurrency. Public reference shows
SMAC3 hits 15.23x vs default 2.25x; the delta is mostly speculative + KV
cache tuning.

### Scenario D (needs D torch baseline first)

D-a. **Balanced launcher** — Average of A and C tuning. `--max-num-seqs 192`,
`mnb-tokens 16384`, chunked prefill on.

### Engine alternatives

E-a. **SGLang launcher for Scenario C** — radix-attention prefix sharing,
overlap scheduling. If vLLM hits a ceiling, SGLang is the next swing.

E-b. **TensorRT-LLM launcher for Scenario C** — requires container with
TensorRT-LLM installed; high upfront cost but largest known speedup ceiling
on H100. Lower priority for RTX PRO 6000 shakedown.

E-c. **Custom torch.compile + paged-attention server** — only meaningful as a
research curio; would only be tried if vLLM/SGLang/TGI all plateau.

## Round 3+ (cross-scenario confirmation)

Take the Round 2 Scenario C winner and run it against A, B, D to compute
`aggregate/geomean_speedup_over_pytorch` matching the public leaderboard
contract. Promote to a mature winner if all four scenarios pass quality and
the geomean speedup exceeds the current advisor-tracked best aggregate.

## Plateau levers (held in reserve)

If multiple consecutive rounds fail to improve on the best Scenario C recipe:

P-a. **Quantization escalation** — fp8 model weights (W8A8 fp8), then int4
AWQ/GPTQ. Requires careful quality gate validation.

P-b. **Engine cocktail** — SGLang for prefill, vLLM for decode, behind a
thin OpenAI-compatible proxy. High implementation cost; only when the
single-engine ceiling is clearly reached.

P-c. **Request reordering** — sort requests by predicted output length to
maximize batch utilization. Requires either a length predictor or an
analytical estimate from prompt features. Speculative because the benchmark
controls input/output lengths; the reordering only helps within a profile.

P-d. **CUDA graph capture custom batch sizes** — by default vLLM captures a
fixed grid; explicitly capturing the exact batch sizes seen in Scenario C
profiles can shave decode latency by removing eager fallback paths.
