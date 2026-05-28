#!/usr/bin/env bash
set -euo pipefail

# Scenario B, frieren PR #166 arm1: n-gram spec depth=25, prompt_lookup_min=1.
# Base: PR #149 winner (vLLM 0.11, BF16, spec=25, lookup_max=12, lookup_min=2) at 3.888x.
# Change: prompt_lookup_min=2 -> 1 (more aggressive lookup, lower precision floor).
# Hypothesis (Rule #15): Sc B's 8192-token output absorbs the precision loss
# that broke Sc D's 2048-token output (PR #154 arm3).

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-16384}"
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

echo "=== vLLM Inference Server (frieren PR #166 arm1: ngram spec 25/12/min=1, Scenario B) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "================================================"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 2048 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --kv-cache-dtype auto \
    --speculative-config '{"method":"ngram","num_speculative_tokens":25,"prompt_lookup_max":12,"prompt_lookup_min":1}' \
    --trust-remote-code \
    --disable-log-stats
