#!/usr/bin/env bash
set -euo pipefail

# Probe for a real SENPAI runtime_env.sh; the launch harness may set
# PROBLEM_DIR=target/ (a relative stub) which breaks the source below.
if [ -z "${PROBLEM_DIR:-}" ] || [ ! -f "${PROBLEM_DIR}/senpai/runtime_env.sh" ]; then
    for _cand in /workspace/senpai-frieren/target /workspace/senpai/target; do
        if [ -f "${_cand}/senpai/runtime_env.sh" ]; then
            PROBLEM_DIR="${_cand}"
            break
        fi
    done
fi
# shellcheck disable=SC1091
source "${PROBLEM_DIR}/senpai/runtime_env.sh"

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"

# Pin FLASH_ATTN.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== vLLM scenario-C high-concurrency throughput ==="
echo "MODEL_ID=${MODEL_ID}  HOST=${HOST}  PORT=${PORT}  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --block-size 16 \
    --trust-remote-code \
    --disable-log-stats
