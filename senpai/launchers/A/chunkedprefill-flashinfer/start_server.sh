#!/usr/bin/env bash
set -euo pipefail

# Scenario A (input-heavy, 8192-token prompts, concurrency=1) launcher.
# Hypothesis: chunked prefill + FlashInfer attention backend lowers TTFT.p50.
#
# Levers:
#   --enable-chunked-prefill            split the long prompt into chunks so
#                                       new requests can decode and CUDA
#                                       graphs stay efficient.
#   --max-num-batched-tokens 16384      cover the 8192-token prompt in one
#                                       chunked-prefill step with headroom.
#   --max-num-seqs 32                   concurrency=1, no need for many seqs.
#   --gpu-memory-utilization 0.92       leave ~8% for OS/driver on the 96 GB
#                                       RTX PRO 6000.
#   VLLM_ATTENTION_BACKEND=FLASHINFER   paged attention with fused page-table
#                                       lookups; usually faster than the
#                                       FLASH_ATTN backend on Blackwell-class
#                                       GPUs for long-context prefill.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# Mistral-7B-Instruct-v0.3 has max_position_embeddings=32768; Scenario A
# prompts are 8192 tokens so 32768 covers the prompt + completion with
# plenty of headroom. Setting MAX_MODEL_LEN above the model limit causes
# vLLM to refuse to start unless VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"
STARTING_VENV_DIR="${INFERENCE_BENCH_STARTING_VENV_DIR:-}"

if [ -n "${STARTING_VENV_DIR}" ] && [ -x "${STARTING_VENV_DIR}/bin/python" ]; then
    export VIRTUAL_ENV="${STARTING_VENV_DIR}"
    export PATH="${STARTING_VENV_DIR}/bin:${PATH}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-${HF_HOME_NEW:-${HOME}/hf_cache}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

# FlashInfer attention backend for long-context prefill. If JIT init fails,
# fall back via VLLM_ATTENTION_BACKEND=FLASH_ATTN (or unset for default).
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"

PY_USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
export PYTHONPATH="/home/agent/task/.local/lib/python3.10/site-packages:${PY_USER_SITE:-}:${PYTHONPATH:-}"

if [ -n "${PROBLEM_DIR:-}" ] && [ -f "${PROBLEM_DIR}/senpai/runtime_env.sh" ]; then
    source "${PROBLEM_DIR}/senpai/runtime_env.sh"
fi

# senpai/runtime_env.sh prepends Python NVIDIA wheel headers (CUDA 12.8) to
# CPATH, but the on-pod nvcc is CUDA 13.2. FlashInfer JIT compilation then
# trips the CCCL "CUDA compiler and CUDA toolkit headers are incompatible"
# guard. Clear CPATH so JIT picks up the matching headers from
# /usr/local/cuda/include directly. LD_LIBRARY_PATH munging is fine to keep
# because the runtime CUDA libs from the wheels are ABI-compatible with what
# vLLM's prebuilt extensions expect.
unset CPATH

echo "=== vLLM Scenario A: chunked prefill + FlashInfer ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "======================================================"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --disable-log-stats \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 32
