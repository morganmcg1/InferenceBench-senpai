#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"
SGLANG_VENV="${SGLANG_VENV:-/tmp/sglang-venv}"

if [ -x "${SGLANG_VENV}/bin/python" ]; then
    export VIRTUAL_ENV="${SGLANG_VENV}"
    export PATH="${SGLANG_VENV}/bin:${PATH}"
fi

export HF_HOME="${HF_HOME:-${HF_HOME_NEW:-${HOME}/hf_cache}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== SGLang (Arm 2: mem-frac 0.85, mrr 128, chunked 8192, fcfs) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "SGLANG_VENV=${SGLANG_VENV}"
echo "=================================================================="

exec python -m sglang.launch_server \
    --model-path "${MODEL_ID}" \
    --host "${HOST}" --port "${PORT}" \
    --context-length "${MAX_MODEL_LEN}" \
    --attention-backend triton \
    --mem-fraction-static 0.85 \
    --max-running-requests 128 \
    --chunked-prefill-size 8192 \
    --schedule-policy fcfs \
    --trust-remote-code
