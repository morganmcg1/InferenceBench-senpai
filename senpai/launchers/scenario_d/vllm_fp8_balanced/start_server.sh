#!/usr/bin/env bash
set -euo pipefail

# Scenario D (balanced, 4096/2048, burst concurrency 4):
# co-optimize TTFT, TPOT, request throughput via FP8 weight + FP8 KV cache,
# moderate chunked prefill, and FlashInfer attention backend on Blackwell.
#
# Geomean of (1/ttft.p50, 1/tpot.p50, request_throughput) penalizes any single
# weak dimension. FP8 weights reduce forward-pass bytes; FP8 KV cache leaves
# headroom for the 4× burst. CUDA graphs default ON for fast decode steps.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-8192}"

if [ -n "${INFERENCE_BENCH_STARTING_VENV_DIR:-}" ] && [ -x "${INFERENCE_BENCH_STARTING_VENV_DIR}/bin/python" ]; then
    export VIRTUAL_ENV="${INFERENCE_BENCH_STARTING_VENV_DIR}"
    export PATH="${INFERENCE_BENCH_STARTING_VENV_DIR}/bin:${PATH}"
fi
export HF_HOME="${HF_HOME:-${HF_HOME_NEW:-${HOME}/hf_cache}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# CUDA pip-package include/library paths — required when the supervised
# relaunch wrapper does not source senpai/runtime_env.sh and vLLM's compiled
# extensions need libcudart.so.12 (CUDA 12 ABI) on a CUDA 13.2 host.
_nvidia_lib_paths="$(python3 - <<'PY'
import site
from pathlib import Path
roots = [Path(p) for p in site.getsitepackages() + [site.getusersitepackages()] if Path(p).exists()]
libs = []
for root in roots:
    nv = root / "nvidia"
    if nv.exists():
        libs.extend(str(p) for p in nv.glob("*/lib") if p.is_dir())
print(":".join(libs))
PY
)"
if [ -n "${_nvidia_lib_paths}" ]; then
    export LD_LIBRARY_PATH="${_nvidia_lib_paths}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/senpai-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${XDG_CACHE_HOME}/cuda}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${XDG_CACHE_HOME}/vllm}"
mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${VLLM_CACHE_ROOT}"

export VLLM_ATTENTION_BACKEND="FLASHINFER"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --block-size 16 \
    --trust-remote-code \
    --disable-log-stats
