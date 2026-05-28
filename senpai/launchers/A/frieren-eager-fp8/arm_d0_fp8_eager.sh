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

# Force-disable FlashInfer sampler/prefill on RTX PRO 6000 image
# (JIT kernel cannot find curand.h; runtime_env defaults overridden upstream).
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER_PREFILL=1
export VLLM_DISABLE_FLASHINFER_SAMPLING=1

# FlashInfer bmm_fp8 JIT linker needs unversioned libcublas.so/libcublasLt.so;
# the nvidia pip package ships only versioned .so.12 files. vLLM picks the
# flashinfer FP8 GEMM path on Blackwell (SM 120 >= 100) via
# vllm/model_executor/layers/quantization/utils/w8a8_utils.py:Fp8LinearOp,
# so without this LIBRARY_PATH the first inference request kills the engine
# with `/usr/bin/ld: cannot find -lcublas`.
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

echo "=== vLLM Inference Server: arm_d0_fp8_eager (PR #169) ==="
echo "MODEL_ID=${MODEL_ID}  --quantization fp8 --enforce-eager"
echo "HOST=${HOST} PORT=${PORT}"
echo "==========================================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --quantization fp8 \
    --kv-cache-dtype auto \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --trust-remote-code \
    --disable-log-stats
