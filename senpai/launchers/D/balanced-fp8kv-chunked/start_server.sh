#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-${HF_HOME_NEW:-${HOME}/hf_cache}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

# Backend selection for sm_120 (RTX PRO 6000 Blackwell):
#   - FLASHINFER backend triggers JIT compile of its sampling kernels and the
#     bundled CCCL/libcudacxx headers are incompatible with the system nvcc.
#   - FLASH_ATTN backend rejects fp8 kv-cache on sm_120 (requires sm_90).
#   - TRITON_ATTN is pure-Triton, has no nvcc/CCCL dependency, and supports
#     fp8 kv-cache on Blackwell, so use it for this fp8-kv + chunked-prefill
#     recipe. Sampling stays on PyTorch-native to avoid the same JIT path.
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0

source "$PROBLEM_DIR/senpai/runtime_env.sh"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --disable-log-stats \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64
