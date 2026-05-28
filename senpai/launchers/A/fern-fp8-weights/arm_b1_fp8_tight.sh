#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
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

# Force-disable FlashInfer sampler/prefill — the JIT kernel cannot find
# curand.h on this RTX PRO 6000 image, and runtime_env defaults are
# overridden somewhere upstream.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER_PREFILL=1
export VLLM_DISABLE_FLASHINFER_SAMPLING=1

echo "=== vLLM Inference Server: arm_b1_fp8_tight (PR #160) ==="
echo "MODEL_ID=${MODEL_ID}  --quantization fp8"
echo "max-model-len=9728 max-num-batched-tokens=16384 max-num-seqs=8"
echo "no-enable-chunked-prefill no-enable-prefix-caching"
echo "HOST=${HOST} PORT=${PORT}"
echo "============================================================"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --quantization fp8 \
    --kv-cache-dtype auto \
    --max-model-len 9728 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.92 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --disable-log-stats
