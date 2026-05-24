#!/usr/bin/env bash
set -euo pipefail

# Scenario A (input-heavy, single-stream long prefill) launcher.
#
# Hypothesis: at concurrency=1 with ~8K prompts, chunked prefill and prefix
# caching add scheduler overhead with no benefit. Run the full prefill in one
# scheduler step with FP8 weights + FP8 KV cache and FlashAttention on Blackwell.

PROBLEM_DIR_FALLBACK="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PROBLEM_DIR="${PROBLEM_DIR:-$PROBLEM_DIR_FALLBACK}"
if [ -f "$PROBLEM_DIR/senpai/runtime_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$PROBLEM_DIR/senpai/runtime_env.sh"
fi

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-131072}"
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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# FlashAttention is the safer Blackwell backend for 8K single-stream prefill.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

echo "=== vLLM Inference Server (scenario-A, FP8 + FlashAttention, no chunked prefill) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "HF_HOME=${HF_HOME}"
echo "===================================================================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 16384 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.88 \
    --block-size 16 \
    --trust-remote-code \
    --disable-log-stats
