#!/usr/bin/env bash
set -euo pipefail

# Scenario A (input-heavy) launcher: FlashInfer attention backend + FP8 KV cache,
# CUDA graphs on, no chunked prefill, no prefix caching, large batched-token
# budget so the 8192-token prefill fits in one efficient kernel call.
#
# Hypothesis: TTFT p50 on Scenario A is prefill-bound at concurrency 1, and the
# unoptimized vLLM default leaves headroom on the attention kernel and KV
# allocation. Switching the attention backend to FlashInfer, removing chunked
# prefill, and dropping KV cache to fp8 should cut TTFT substantially while
# preserving MMLU-Pro quality.
#
# Foreground exec, no daemonization, no hardcoded HOST/PORT/MODEL_ID.

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

# Attention backend. FLASHINFER is the original target; on this Blackwell pod
# FlashInfer's bundled CUDA 12 headers conflict with system CUDA 13 nvcc, so
# FLASH_ATTN is the override path used in practice. FLASH_ATTN on Blackwell does
# not support fp8 kv-cache, so the KV dtype is overrideable too.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

# Disable FlashInfer sampler — vLLM auto-enables it when flashinfer is importable
# but its sampling kernel JIT-build fails on this CUDA 13 / Blackwell pod
# (curand.h missing in /usr/local/cuda/include). PyTorch sampler is fine for
# concurrency-1 input-heavy workloads where sampling is not the bottleneck.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

echo "=== vLLM Inference Server (Scenario A: tuned prefill batch) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "HF_HOME=${HF_HOME}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
echo "================================================"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --max-num-seqs 16 \
    --max-num-batched-tokens 16384 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --disable-log-stats
