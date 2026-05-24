#!/usr/bin/env bash
set -euo pipefail

# Scenario A (input-heavy TTFT) launcher.
#
# Combines FP8 weights + FP8 KV with the FlashInfer attention backend and
# disables chunked prefill so a full 8192-token prompt at burst c=1 executes
# in a single iteration. Optimizes 1 / ttft.p50 for Mistral-7B-Instruct-v0.3
# on a single Blackwell-class GPU.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-12288}"
STARTING_VENV_DIR="${INFERENCE_BENCH_STARTING_VENV_DIR:-}"
KV_CACHE_DTYPE="${SENPAI_KV_CACHE_DTYPE:-fp8}"

if [ -n "${STARTING_VENV_DIR}" ] && [ -x "${STARTING_VENV_DIR}/bin/python" ]; then
    export VIRTUAL_ENV="${STARTING_VENV_DIR}"
    export PATH="${STARTING_VENV_DIR}/bin:${PATH}"
fi

if [ -n "${PROBLEM_DIR:-}" ] && [ -f "${PROBLEM_DIR}/senpai/runtime_env.sh" ]; then
    # shellcheck disable=SC1091
    source "${PROBLEM_DIR}/senpai/runtime_env.sh"
fi

export HF_HOME="${HF_HOME:-${HF_HOME_NEW:-${HOME}/hf_cache}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

PY_USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
export PYTHONPATH="/home/agent/task/.local/lib/python3.10/site-packages:${PY_USER_SITE:-}:${PYTHONPATH:-}"

# Force integer device index — some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Must be exported BEFORE vLLM imports its attention backends.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"

# FlashInfer JIT compiles some .cu kernels at first use (e.g. sampling/curand).
# nvcc's CUDA frontend ignores CPATH for device-side header lookups. The pod
# ships nvcc 13.2 but the CUDA 13 toolkit no longer bundles curand/cublas/
# cusolver headers — those live only in the nvidia/* python wheels (12.8).
# We prepend -I for only the math-lib wheels so nvcc keeps its own
# cuda_runtime headers (CUDART_VERSION 13.x matches nvcc, satisfying CCCL's
# version check). Pulling nvidia/cuda_runtime in here would otherwise break
# the build with "CUDA compiler and CUDA toolkit headers are incompatible".
_math_lib_includes="$(python3 - <<'PYI'
import site
from pathlib import Path
MATH_LIBS = {
    "curand", "cublas", "cusolver", "cufft", "cusparse", "cusparselt",
    "nccl", "nvtx", "nvjitlink", "cudnn", "cufile",
}
incs = []
seen = set()
for root in site.getsitepackages() + [site.getusersitepackages()]:
    base = Path(root) / "nvidia"
    if not base.exists():
        continue
    for pkg in MATH_LIBS:
        inc = base / pkg / "include"
        if inc.is_dir():
            sinc = str(inc)
            if sinc not in seen:
                seen.add(sinc)
                incs.append(sinc)
print(" ".join(f"-I{i}" for i in incs))
PYI
)"
if [ -n "${_math_lib_includes}" ]; then
    export NVCC_PREPEND_FLAGS="${_math_lib_includes}${NVCC_PREPEND_FLAGS:+ ${NVCC_PREPEND_FLAGS}}"
fi

echo "=== vLLM scA FP8 + FlashInfer prefill launcher ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "HF_HOME=${HF_HOME}"
echo "==================================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --quantization fp8 \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --max-num-seqs 16 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.92 \
    --block-size 16 \
    --no-enable-chunked-prefill \
    --enable-prefix-caching \
    --trust-remote-code \
    --disable-log-stats
