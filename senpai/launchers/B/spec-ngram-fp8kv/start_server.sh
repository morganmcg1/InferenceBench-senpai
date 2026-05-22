#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-10240}"
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
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
export VLLM_NO_USAGE_STATS=1

NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"
PROMPT_LOOKUP_MAX="${PROMPT_LOOKUP_MAX:-4}"
PROMPT_LOOKUP_MIN="${PROMPT_LOOKUP_MIN:-2}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-{\"method\":\"ngram\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS},\"prompt_lookup_max\":${PROMPT_LOOKUP_MAX},\"prompt_lookup_min\":${PROMPT_LOOKUP_MIN}}}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --tokenizer-mode auto \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 10240 \
    --block-size 16 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --kv-cache-dtype fp8 \
    --speculative-config "${SPECULATIVE_CONFIG}" \
    --trust-remote-code \
    --disable-log-stats
