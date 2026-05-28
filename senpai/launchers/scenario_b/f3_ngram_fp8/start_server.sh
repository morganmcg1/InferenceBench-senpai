#!/usr/bin/env bash
# Scenario B arm F3 — winning n-gram spec config + FP8 weight quantization.
# Stacks bandwidth-side FP8 weights with decode-side speculation. KV cache
# stays BF16 because the RTX PRO 6000 FlashAttention path rejects FP8 KV.
#
# IMPORTANT: this launcher is templated to use F1 (k=5) settings by default.
# Override INFERENCE_BENCH_SPEC_CONFIG to swap to F2 (k=3) if F2 wins F1.
set -euo pipefail

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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== Scenario B / F3 n-gram k=5 + FP8 weights ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --disable-log-stats \
    --quantization fp8 \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'
