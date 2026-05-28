#!/usr/bin/env bash
set -euo pipefail
MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
export HF_HOME="${HF_HOME:-${HF_HOME_NEW:-${HOME}/hf_cache}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
PY_USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
export PYTHONPATH="/home/agent/task/.local/lib/python3.10/site-packages:${PY_USER_SITE:-}:${PYTHONPATH:-}"

# FlashInfer prefill enabled, sampler still on PyTorch path (curand.h JIT broken).
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_DISABLE_FLASHINFER_PREFILL=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER_SAMPLING=1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== vLLM: arm_e0 FlashInfer + FP8 + enforce_eager ==="
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" --host "$HOST" --port "$PORT" \
    --quantization fp8 \
    --kv-cache-dtype auto \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --trust-remote-code \
    --disable-log-stats
