#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-8192}"

if [ -n "${PROBLEM_DIR:-}" ] && [ -f "${PROBLEM_DIR}/senpai/runtime_env.sh" ]; then
  # shellcheck disable=SC1091
  source "${PROBLEM_DIR}/senpai/runtime_env.sh"
fi

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== vLLM launcher: Scenario C / high-throughput / no spec / BF16 KV ==="
echo "MODEL_ID=${MODEL_ID}  HOST=${HOST}  PORT=${PORT}  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "======================================================================"

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs 256 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.95 \
  --block-size 16 \
  --kv-cache-dtype auto \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --trust-remote-code \
  --disable-log-stats
