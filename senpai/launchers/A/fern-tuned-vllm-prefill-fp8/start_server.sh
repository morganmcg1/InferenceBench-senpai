#!/usr/bin/env bash
set -euo pipefail

# Scenario A tuned vLLM prefill launcher — arm 3, FP8 weight quantization.
#
# Identical to fern-tuned-vllm-prefill-noprefix (arm 2) but adds
# `--quantization fp8`. vLLM applies on-the-fly FP8 weight quantization to
# mistralai/Mistral-7B-Instruct-v0.3; the base model is preserved and only the
# weights are 8-bit. KV cache remains BF16 (do NOT add --kv-cache-dtype fp8 —
# that combination does not boot with FLASH_ATTN on this hardware).
#
# Inherits process-local CUDA paths and FlashInfer disablement from
# senpai/runtime_env.sh, which must be sourced by the supervising harness.

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

echo "=== vLLM Inference Server (fern Scenario A tuned prefill, FP8 weights) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=12288 MAX_NUM_BATCHED_TOKENS=16384 MAX_NUM_SEQS=16"
echo "GPU_MEM_UTIL=0.95 PREFIX_CACHING=off CHUNKED_PREFILL=off QUANTIZATION=fp8"
echo "HF_HOME=${HF_HOME}"
echo "============================================================================"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len 12288 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.95 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --quantization fp8 \
    --trust-remote-code \
    --disable-log-stats
