#!/usr/bin/env bash
# Scenario D balanced launcher: chunked prefill + CUDA graphs +
# n-gram speculative decoding (vLLM 0.11.0).
#
# Rationale (PR #81): at concurrency 4 / 4K-in / 2K-out both prefill and
# decode matter. Chunked prefill with a generous --max-num-batched-tokens
# lets decode iterations slip alongside prefill chunks. n-gram speculative
# decoding helps the ~2K-token decode tail because repeated prompt n-grams
# (instructions, system text) are very likely to recur in the answer.

set -euo pipefail

# Allow either PROBLEM_DIR (helper-created workspaces) or default to /workspace.
RUNTIME_ENV="${PROBLEM_DIR:-/workspace/senpai-tanjiro/target}/senpai/runtime_env.sh"
if [ -f "$RUNTIME_ENV" ]; then
    # shellcheck disable=SC1090
    source "$RUNTIME_ENV"
fi

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# 4K in + 2K out fits in 16K; smaller max_model_len gives more KV headroom.
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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

echo "=== vLLM Scenario D balanced launcher (tanjiro_balanced_specdec) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "================================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --block-size 32 \
    --speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
    --trust-remote-code \
    --disable-log-stats
