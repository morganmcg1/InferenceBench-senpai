#!/usr/bin/env bash
set -euo pipefail
MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"; PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-16384}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" --host "${HOST}" --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --kv-cache-dtype auto \
  --quantization fp8 \
  --trust-remote-code --disable-log-stats
