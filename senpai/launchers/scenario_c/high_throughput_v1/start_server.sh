#!/usr/bin/env bash
set -euo pipefail

# SENPAI Scenario C high-throughput vLLM launcher (PR #4, r1-tanjiro/sc-c-high-throughput).
#
# Hypothesis: Scenario C primary metric is geomean req/s across burst / poisson /
# constant profiles. Decode concurrency is the bottleneck, so:
#   - large --max-num-seqs keeps many decode streams active
#   - --enable-chunked-prefill + large --max-num-batched-tokens prevents
#     prefill stalls
#   - --gpu-memory-utilization 0.95 maximizes KV-cache size
#   - --enable-prefix-caching reuses shared system+instruction prefixes
#   - FP8 KV cache doubles effective KV memory (C2 arm only)
#   - FLASH_ATTN backend is the most reliable throughput backend
#
# Per the program contract this script keeps the final server in the foreground
# (exec) so the supervised relaunch harness can manage it.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"
STARTING_VENV_DIR="${INFERENCE_BENCH_STARTING_VENV_DIR:-}"

# Tunable knobs (arm-controlled). Defaults match arm C1; environment overrides
# implement C2 (fp8 KV, max-num-seqs=512) and C3 (prefix caching off).
MAX_NUM_SEQS="${SENPAI_MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${SENPAI_MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEM_UTIL="${SENPAI_GPU_MEM_UTIL:-0.95}"
KV_CACHE_DTYPE="${SENPAI_KV_CACHE_DTYPE:-auto}"
PREFIX_CACHING="${SENPAI_PREFIX_CACHING:-on}"

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

# vLLM 0.21 ships against torch 2.11 (CUDA 12 ABI). The system CUDA toolkit on
# this pod is 13.2, so libcudart.so.12 is only available in the pip-installed
# nvidia cuda_runtime wheel. Stitch torch's bundled libs and that wheel onto
# LD_LIBRARY_PATH so vLLM's compiled extensions resolve correctly.
TORCH_LIB_DIR="$(python3 -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null || true)"
CUDA_RT_DIR="$(python3 -c 'import pathlib, importlib.util; spec = importlib.util.find_spec("nvidia.cuda_runtime"); print(str(pathlib.Path(spec.submodule_search_locations[0]) / "lib")) if spec else None' 2>/dev/null || true)"
if [ -n "${TORCH_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi
if [ -n "${CUDA_RT_DIR}" ]; then
    export LD_LIBRARY_PATH="${CUDA_RT_DIR}:${LD_LIBRARY_PATH:-}"
fi

# FlashAttention is the most reliable throughput backend for Scenario C on
# H100/Blackwell. vLLM will compile FlashInfer kernels for FP8 KV when needed.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

# This pod ships CUDA 13.2 runtime but no libcurand-dev headers, so FlashInfer's
# JIT compile of the sampling kernels fails with `curand.h: No such file or
# directory`. Disable FlashInfer sampling and fall back to vLLM's native
# top-p/top-k sampler, which has no special perf advantage at temperature 0.3
# without custom top-k/top-p.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# Force integer device index — some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PREFIX_FLAG=()
if [[ "${PREFIX_CACHING}" == "on" ]]; then
    PREFIX_FLAG+=(--enable-prefix-caching)
fi

echo "=== vLLM Inference Server (sc-c high-throughput v1) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEM_UTIL}"
echo "KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
echo "PREFIX_CACHING=${PREFIX_CACHING}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "HF_HOME=${HF_HOME}"
echo "================================================"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --enable-chunked-prefill \
    "${PREFIX_FLAG[@]}" \
    --trust-remote-code \
    --disable-log-stats
