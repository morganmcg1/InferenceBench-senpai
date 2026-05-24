#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-8192}"
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
# FlashInfer 0.6.11 JIT requires CUDA 12 toolchain; this pod ships nvcc 13.2,
# so flashinfer kernels fail to build for sm_120 (RTX PRO 6000 Blackwell, CC
# 12.0). FlashAttention v3 only supports fp8 KV cache on Hopper (CC 9.x), so
# fall back to the TRITON_ATTN backend which supports fp8 KV on RTX PRO 6000.
# Also disable the flashinfer sampler so vLLM stays on the torch-native
# sampling path.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_DISABLE_FLASHINFER_PREFILL="${VLLM_DISABLE_FLASHINFER_PREFILL:-1}"

echo "=== vLLM Scenario C launcher: FP8 W8A8 + FP8 KV + TRITON_ATTN (flashinfer disabled) ==="
echo "MODEL_ID=${MODEL_ID} HOST=${HOST} PORT=${PORT} MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "ATTN_BACKEND=${VLLM_ATTENTION_BACKEND} SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER} PREFILL_DISABLE=${VLLM_DISABLE_FLASHINFER_PREFILL}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --trust-remote-code \
    --disable-log-stats
