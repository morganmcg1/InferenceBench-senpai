#!/usr/bin/env bash
set -euo pipefail

# Scenario A (input-heavy, 8K input / 1K output, burst concurrency 1).
# Goal: minimise TTFT.p50 by sizing the per-step batched-token budget to
# cover the full 8K prompt, enabling chunked prefill, using FP8 KV cache,
# and keeping CUDA graphs active for the small post-prefill decode steps.
# Do NOT re-enable FlashInfer sampler/prefill — senpai/runtime_env.sh
# disables those on the RTX PRO 6000 shakedown hardware.

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
    --max-num-batched-tokens 16384 \
    --max-num-seqs 32 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8 \
    --block-size 32 \
    --tokenizer-mode auto \
    --disable-log-stats \
    --trust-remote-code
