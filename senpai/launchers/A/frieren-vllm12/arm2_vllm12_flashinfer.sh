#!/usr/bin/env bash
# Sc A frieren PR #186 arm2: vLLM 0.21 (latest 0.12+) + FlashInfer attention backend.
# Target arm: tests whether FlashInfer now runs on SM120 Blackwell with vLLM 0.21.
set -euo pipefail

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-16384}"
VLLM12_VENV="/tmp/inferencebench-engine-venvs/vllm12-pr-186"

if [ ! -x "${VLLM12_VENV}/bin/python" ]; then
    echo "[arm2] Installing vLLM 0.12+ in ${VLLM12_VENV} ..."
    if command -v uv >/dev/null 2>&1; then
        uv venv "${VLLM12_VENV}" --python 3.10 --seed
        "${VLLM12_VENV}/bin/pip" install --upgrade pip wheel setuptools
        "${VLLM12_VENV}/bin/pip" install "vllm>=0.12.0"
    else
        python3.10 -m venv "${VLLM12_VENV}"
        "${VLLM12_VENV}/bin/pip" install --upgrade pip wheel setuptools
        "${VLLM12_VENV}/bin/pip" install "vllm>=0.12.0"
    fi
fi

export VIRTUAL_ENV="${VLLM12_VENV}"
export PATH="${VLLM12_VENV}/bin:${PATH}"

export HF_HOME="${HF_HOME:-${HOME}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Sampler stays standard. Enable FlashInfer prefill (runtime_env.sh defaults
# block it for backwards compat) and route the backend through FLASHINFER.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER_PREFILL=0

echo "=== Sc A arm2: vLLM $(${VLLM12_VENV}/bin/python -c 'import vllm; print(vllm.__version__)') + FlashInfer ==="

exec "${VLLM12_VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 16384 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --kv-cache-dtype auto \
    --quantization fp8 \
    --attention-backend FLASHINFER \
    --trust-remote-code \
    --disable-log-stats
