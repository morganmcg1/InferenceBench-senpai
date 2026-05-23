#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# Sc. A inputs are <=8192 tokens, outputs <=1024; set max_model_len to cover both.
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-12288}"
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
# vLLM 0.21 uses the --attention-backend CLI flag (VLLM_ATTENTION_BACKEND env was removed).
# We still export it as a hint for any tooling that reads it.
export VLLM_ATTENTION_BACKEND="FLASHINFER"
# FlashInfer sampling JIT needs curand.h, which is missing on this pod
# (libcurand-13-2 runtime is installed but no dev headers package is available
# via apt for cuda 13.2). Disabling FlashInfer sampling falls back to the
# PyTorch-native sampler; FlashInfer attention kernels do NOT depend on curand
# (only flashinfer/data/include/flashinfer/sampling.cuh references it) so the
# primary FlashInfer attention lever is preserved.
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "=== vLLM Scenario A: FlashInfer + chunked prefill + FP8 KV + prefix caching ==="
echo "MODEL_ID=${MODEL_ID} HOST=${HOST} PORT=${PORT} MAX_MODEL_LEN=${MAX_MODEL_LEN}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --attention-backend FLASHINFER \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 16384 \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --block-size 16 \
    --trust-remote-code \
    --disable-log-stats
