#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-16384}"
STARTING_VENV_DIR="${INFERENCE_BENCH_STARTING_VENV_DIR:-}"

if [ -n "${STARTING_VENV_DIR}" ] && [ -x "${STARTING_VENV_DIR}/bin/python" ]; then
    export VIRTUAL_ENV="${STARTING_VENV_DIR}"
    export PATH="${STARTING_VENV_DIR}/bin:${PATH}"
fi

if [ -n "${PROBLEM_DIR:-}" ] && [ -f "$PROBLEM_DIR/senpai/runtime_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$PROBLEM_DIR/senpai/runtime_env.sh"
fi

export HF_HOME="${HF_HOME:-${HF_HOME_NEW:-${HOME}/hf_cache}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

PY_USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
export PYTHONPATH="/home/agent/task/.local/lib/python3.10/site-packages:${PY_USER_SITE:-}:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Advisor Blackwell finding (RTX PRO 6000, sm_120, CUDA 13.2):
#   - FlashInfer JIT fails to build for sm_120 → cannot use FLASHINFER backend.
#   - FlashAttention v3 fp8 KV path is Hopper-only → FLASH_ATTN + --kv-cache-dtype fp8 errors.
# → Default to TRITON_ATTN with FlashInfer sampler/prefill disabled.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_DISABLE_FLASHINFER_PREFILL="${VLLM_DISABLE_FLASHINFER_PREFILL:-1}"

echo "=== vLLM Scenario B launcher: FP8 W8A8 + FP8 KV + n-gram speculative ==="
echo "MODEL_ID=${MODEL_ID} HOST=${HOST} PORT=${PORT} MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "ATTN_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER} VLLM_DISABLE_FLASHINFER_PREFILL=${VLLM_DISABLE_FLASHINFER_PREFILL}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 4096 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
    --trust-remote-code \
    --disable-log-stats
