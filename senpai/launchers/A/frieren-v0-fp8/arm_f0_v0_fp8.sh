#!/usr/bin/env bash
set -euo pipefail

# Scenario A — Arm F0: FP8 weights + V0 engine (legacy dispatch).
#
# Based on D0 (PR #169 / FP8 winner) but with --enforce-eager removed and
# VLLM_USE_V1=0 to opt into the legacy V0 engine path. Tests whether V0's
# simpler dispatch beats V1 for single-concurrency long-prefill on this
# image. cublas symlink fix carried verbatim from D0; required because
# vLLM picks the FlashInfer FP8 GEMM path on this Blackwell SM 120 GPU.

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

# Sampler/prefill FlashInfer disabled (image-level constraint:
# JIT kernel cannot find curand.h on the RTX PRO 6000 image).
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER_PREFILL=1
export VLLM_DISABLE_FLASHINFER_SAMPLING=1

# Engine version: opt out of V1 to use the legacy V0 engine path.
# This is the single new toggle vs the D0 FP8 winner.
export VLLM_USE_V1=0

# cublas symlink workaround (carried from D0). FlashInfer bmm_fp8 JIT linker
# needs unversioned libcublas.so / libcublasLt.so; nvidia pip package ships
# only versioned .so.12 files. vLLM picks the flashinfer FP8 GEMM path on
# Blackwell (SM 120 >= 100), so without this LIBRARY_PATH the first inference
# request kills the engine with `/usr/bin/ld: cannot find -lcublas`.
CUBLAS_STUB_DIR="${CUBLAS_STUB_DIR:-/tmp/inferencebench-cublas-stubs}"
mkdir -p "$CUBLAS_STUB_DIR"
NVIDIA_CUBLAS_LIB="/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib"
if [ -f "$NVIDIA_CUBLAS_LIB/libcublas.so.12" ]; then
    ln -sf "$NVIDIA_CUBLAS_LIB/libcublas.so.12" "$CUBLAS_STUB_DIR/libcublas.so"
fi
if [ -f "$NVIDIA_CUBLAS_LIB/libcublasLt.so.12" ]; then
    ln -sf "$NVIDIA_CUBLAS_LIB/libcublasLt.so.12" "$CUBLAS_STUB_DIR/libcublasLt.so"
fi
export LIBRARY_PATH="${CUBLAS_STUB_DIR}${LIBRARY_PATH:+:${LIBRARY_PATH}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== vLLM: arm_f0 FP8 weights + V0 engine (PR #176) ==="
echo "MODEL_ID=${MODEL_ID}  --quantization fp8  VLLM_USE_V1=${VLLM_USE_V1}"
echo "HOST=${HOST} PORT=${PORT}"
echo "======================================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --quantization fp8 \
    --kv-cache-dtype auto \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --disable-log-stats
