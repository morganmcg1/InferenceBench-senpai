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

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER_PREFILL=1
export VLLM_DISABLE_FLASHINFER_SAMPLING=1

mkdir -p /tmp/inferencebench-cublas-stubs
NV_CUBLAS_DIR="$(python3 -c 'import os, site; \
  paths=[os.path.join(p, "nvidia/cublas/lib") for p in site.getsitepackages()+[site.getusersitepackages()]]; \
  print(next((p for p in paths if os.path.isdir(p)), ""))' 2>/dev/null || true)"
if [ -n "$NV_CUBLAS_DIR" ] && [ -d "$NV_CUBLAS_DIR" ]; then
  ln -sf "$NV_CUBLAS_DIR/libcublas.so.12"   /tmp/inferencebench-cublas-stubs/libcublas.so   2>/dev/null || true
  ln -sf "$NV_CUBLAS_DIR/libcublasLt.so.12" /tmp/inferencebench-cublas-stubs/libcublasLt.so 2>/dev/null || true
  export LIBRARY_PATH="/tmp/inferencebench-cublas-stubs:${LIBRARY_PATH:-}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== vLLM: arm_g0 FP8 weights + max-num-batched-tokens=32768 ==="
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" --host "$HOST" --port "$PORT" \
    --quantization fp8 \
    --kv-cache-dtype auto \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 32768 \
    --trust-remote-code \
    --disable-log-stats
