#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"

SGLANG_VENV="${SGLANG_VENV:-/tmp/inferencebench-engine-venvs/sglang-pr-163}"
if [ ! -x "${SGLANG_VENV}/bin/python" ]; then
    echo "ERROR: SGLang venv not found at ${SGLANG_VENV}" >&2
    echo "Recreate with: python \"\${PROBLEM_DIR}/senpai/create_engine_venv.py\" --engine sglang --pr 163" >&2
    exit 1
fi
source "${SGLANG_VENV}/bin/activate"

export HF_HOME="${HF_HOME:-${HF_HOME_NEW:-${HOME}/hf_cache}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ATTN_BACKEND="${SGLANG_ATTENTION_BACKEND:-triton}"
SAMPLING_BACKEND="${SGLANG_SAMPLING_BACKEND:-pytorch}"
MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.88}"
MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-256}"
SCHEDULE_POLICY="${SGLANG_SCHEDULE_POLICY:-fcfs}"
CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-4096}"
EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"

echo "=== SGLang Inference Server (scen-c-fern) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "ATTENTION_BACKEND=${ATTN_BACKEND}"
echo "SAMPLING_BACKEND=${SAMPLING_BACKEND}"
echo "MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC}"
echo "MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}"
echo "SCHEDULE_POLICY=${SCHEDULE_POLICY}"
echo "CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE}"
echo "EXTRA_ARGS=${EXTRA_ARGS}"
echo "VENV=${SGLANG_VENV}"
echo "================================================"

args=(
    --model-path "${MODEL_ID}"
    --host "${HOST}" --port "${PORT}"
    --context-length "${MAX_MODEL_LEN}"
    --attention-backend "${ATTN_BACKEND}"
    --sampling-backend "${SAMPLING_BACKEND}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --trust-remote-code
)

if [ -n "${MAX_RUNNING_REQUESTS}" ]; then
    args+=(--max-running-requests "${MAX_RUNNING_REQUESTS}")
fi
if [ -n "${SCHEDULE_POLICY}" ]; then
    args+=(--schedule-policy "${SCHEDULE_POLICY}")
fi
if [ -n "${CHUNKED_PREFILL_SIZE}" ]; then
    args+=(--chunked-prefill-size "${CHUNKED_PREFILL_SIZE}")
fi
if [ -n "${EXTRA_ARGS}" ]; then
    read -r -a extra <<< "${EXTRA_ARGS}"
    args+=("${extra[@]}")
fi

exec python -m sglang.launch_server "${args[@]}"
