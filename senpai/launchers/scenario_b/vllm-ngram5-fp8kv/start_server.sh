#!/usr/bin/env bash
set -euo pipefail

# PR #76 launcher: vLLM + n-gram speculative decoding (5 tokens) for Scenario B.
#
# NOTE on FP8 KV cache: the PR's original config asked for `--kv-cache-dtype fp8`,
# but on the RTX PRO 6000 (Blackwell) shakedown hardware vLLM 0.11.0's FLASH_ATTN
# backend rejects fp8 kv-cache with "FlashAttention does not support fp8 kv-cache
# on this device." That fp8 kv path appears to be H100/SM90-only in this vLLM
# build. The PR also forbids re-enabling FlashInfer on this hardware, so we drop
# fp8 kv-cache and keep everything else (n-gram spec=5 decoding, FLASH_ATTN,
# block_size 16, max_num_seqs 8, no chunked prefill, no prefix caching). With
# ~96GB VRAM on this device, the memory headroom that fp8 kv was supposed to
# free is not needed for graph capture + speculative buffers at concurrency 1.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_NO_USAGE_STATS=1

exec python3 -u -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --host "${HOST}" --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --block-size 16 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --tokenizer-mode auto \
  --disable-log-stats \
  --trust-remote-code \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'
