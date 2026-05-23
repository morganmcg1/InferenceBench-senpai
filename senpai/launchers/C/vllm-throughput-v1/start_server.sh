#!/usr/bin/env bash
set -euo pipefail

# Scenario C (high-load throughput) vLLM launcher v1.
#
# Foreground server only; the benchmark harness supervises launch and teardown.
# Source senpai/runtime_env.sh first so CPATH / LD_LIBRARY_PATH / cache dirs are
# set before vLLM imports CUDA pip-package extensions.

if [ -n "${PROBLEM_DIR:-}" ] && [ -f "${PROBLEM_DIR}/senpai/runtime_env.sh" ]; then
    # shellcheck disable=SC1091
    source "${PROBLEM_DIR}/senpai/runtime_env.sh"
elif [ -f "$(dirname "$0")/../../../runtime_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$(dirname "$0")/../../../runtime_env.sh"
fi

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"
STARTING_VENV_DIR="${INFERENCE_BENCH_STARTING_VENV_DIR:-}"

if [ -n "${STARTING_VENV_DIR}" ] && [ -x "${STARTING_VENV_DIR}/bin/python" ]; then
    export VIRTUAL_ENV="${STARTING_VENV_DIR}"
    export PATH="${STARTING_VENV_DIR}/bin:${PATH}"
fi

export HF_HOME="${HF_HOME:-${HF_HOME_NEW:-${HOME}/hf_cache}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

PY_USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
export PYTHONPATH="/home/agent/task/.local/lib/python3.10/site-packages:${PY_USER_SITE:-}:${PYTHONPATH:-}"

# Force integer device index — some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== vLLM Scenario C Throughput-Tuned Launcher v1 ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "HF_HOME=${HF_HOME}"
echo "Tuned for Scenario C (1024-in/1024-out, burst c=64, poisson 32 r/s, constant 16 r/s)"
echo "===================================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs 256 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.92 \
    --no-enable-prefix-caching \
    --enable-chunked-prefill \
    --kv-cache-dtype auto \
    --tokenizer-mode auto \
    --trust-remote-code \
    --disable-log-stats
