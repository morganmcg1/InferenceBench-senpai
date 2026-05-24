# scA / vllm-fp8-flashinfer-prefill

Scenario A (input-heavy TTFT) launcher. Single 8192-token burst c=1 prefill is
dominated by attention compute, so this recipe combines:

- `VLLM_ATTENTION_BACKEND=FLASHINFER` — fastest available prefill kernel on
  Hopper/Blackwell when it loads cleanly. vLLM silently falls back to
  TRITON_ATTN on Blackwell if FlashInfer validation fails; confirm the
  selected backend in server startup logs.
- `--quantization fp8 --kv-cache-dtype fp8` — on-the-fly FP8 weight cast +
  FP8 e4m3 KV cache. Roughly doubles prefill matmul throughput vs. FP16 on
  Blackwell.
- `--no-enable-chunked-prefill` — burst c=1 has no other prompts to
  interleave with, so chunked prefill only adds per-iteration overhead.
- `--max-num-batched-tokens 16384` — keeps the full 8192-token prompt in a
  single iteration token budget.
- `--enable-prefix-caching` — cheap insurance against repeated identical
  prefixes; harmless when prompts differ.
- `--max-num-seqs 16`, `--gpu-memory-utilization 0.92`, `--block-size 16`,
  no `--enforce-eager` — defaults tuned for the Scenario A burst c=1
  workload while leaving room for CUDA graph capture.

Tracked in PR #57.
