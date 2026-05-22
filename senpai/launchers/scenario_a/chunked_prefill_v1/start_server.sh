#!/usr/bin/env bash
set -euo pipefail

# SENPAI launcher: Scenario A (input-heavy) — vLLM with chunked prefill
# and a larger per-step batched-token budget.
#
# Hypothesis (see PR #2): Scenario A is prefill-dominated (8192 input tokens,
# concurrency 1). Raising --max-num-batched-tokens shortens prefill into fewer
# scheduler iterations and reduces TTFT. Optional FP8 KV cache frees memory
# headroom while keeping all weights in fp16.
#
# Knobs exposed via environment variables (defaults are the winning arm):
#   SENPAI_VLLM_MAX_NUM_BATCHED_TOKENS  default: 16384 (A1/A2) or 32768 (A3)
#   SENPAI_VLLM_MAX_NUM_SEQS            default: 16
#   SENPAI_VLLM_KV_CACHE_DTYPE          default: auto (A1/A3) or fp8 (A2)
#   SENPAI_VLLM_GPU_MEMORY_UTILIZATION  default: 0.90
#   SENPAI_VLLM_ATTENTION_BACKEND       default: FLASH_ATTN
#
# Foreground server (exec). Do not daemonize: the official benchmark
# supervises start_server.sh under launch_supervised_server.sh and restarts
# it in a fresh container for final scoring.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"
STARTING_VENV_DIR="${INFERENCE_BENCH_STARTING_VENV_DIR:-}"

MAX_NUM_BATCHED_TOKENS="${SENPAI_VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${SENPAI_VLLM_MAX_NUM_SEQS:-16}"
KV_CACHE_DTYPE="${SENPAI_VLLM_KV_CACHE_DTYPE:-auto}"
GPU_MEMORY_UTILIZATION="${SENPAI_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
ATTENTION_BACKEND="${SENPAI_VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

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

# Ensure vLLM's _C extension can find libtorch.* and libcudart.so.* (no-op when
# the container's default LD search path already covers them).
TORCH_LIB_DIR="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null || true)"
NVIDIA_CUDART_LIB_DIR="$(python3 -c $'import glob\nmatches = sorted(glob.glob("/usr/local/lib/python3.*/dist-packages/nvidia/cuda_runtime/lib")) + sorted(glob.glob("/usr/lib/python3/*/site-packages/nvidia/cuda_runtime/lib"))\nprint(matches[0] if matches else "")' 2>/dev/null || true)"
if [ -n "${TORCH_LIB_DIR}" ]; then
    export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi
if [ -n "${NVIDIA_CUDART_LIB_DIR}" ] && [ "${NVIDIA_CUDART_LIB_DIR}" != "None" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_CUDART_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

# Force integer device index — some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# vLLM 0.21 dropped the VLLM_ATTENTION_BACKEND env var and now prints
# "Unknown vLLM environment variable detected" if it is set. The
# --attention-backend CLI flag below is the supported path on this pod.
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
# FlashInfer's top-p/top-k sampler JITs curand into a sm_120 kernel for
# Blackwell. The container ships libcurand runtime but not the dev headers,
# so JIT compilation fails. Disable the FlashInfer sampler to fall back to
# the native PyTorch sampler — purely a sampling path change, not the
# attention path, so it does not affect TTFT/TPOT behavior we are tuning.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

echo "=== vLLM Inference Server (chunked_prefill_v1 for Scenario A) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "ATTENTION_BACKEND=${ATTENTION_BACKEND}"
echo "HF_HOME=${HF_HOME}"
echo "==============================================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --attention-backend "${ATTENTION_BACKEND}" \
    --enable-chunked-prefill \
    --trust-remote-code \
    --disable-log-stats
