#!/usr/bin/env bash
set -euo pipefail

# Scenario A tuned vLLM prefill launcher.
#
# Mistral-7B-Instruct-v0.3, RTX PRO 6000, burst concurrency 1, 8192-in / 1024-out.
# Goal: minimize TTFT.p50 by letting the full ~8k prefill fit in one forward
# pass and shrinking KV-cache reservation to free room for batched tokens.
#
# Inherits process-local CUDA paths and FlashInfer disablement from
# senpai/runtime_env.sh, which must be sourced by the supervising harness.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
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

echo "=== vLLM Inference Server (fern Scenario A tuned prefill) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=12288 MAX_NUM_BATCHED_TOKENS=16384 MAX_NUM_SEQS=16"
echo "GPU_MEM_UTIL=0.95 PREFIX_CACHING=on CHUNKED_PREFILL=off"
echo "HF_HOME=${HF_HOME}"
echo "================================================="

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len 12288 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --no-enable-chunked-prefill \
    --trust-remote-code \
    --disable-log-stats
