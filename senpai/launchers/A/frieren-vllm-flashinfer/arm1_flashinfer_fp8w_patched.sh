#!/usr/bin/env bash
set -euo pipefail

# Sc A arm1-patched (PR #148): FlashInfer attention + FP8 weights, with a
# compat shim that fixes vLLM 0.11 <-> FlashInfer 0.6+ positional-arg drift.
#
# Without the shim, BatchDecodeWithPagedKVCacheWrapper.plan() receives
# sm_scale=None because FlashInfer 0.6 inserted an extra positional parameter,
# and vLLM's forward() trips `assert decode_wrapper._sm_scale == self.scale`.
# The shim (vllm_flashinfer_compat_patch.py) wraps .plan() to re-dispatch
# positional args as keyword args. We load it via sitecustomize.py so vLLM's
# spawned engine workers also see the patch (without sitecustomize the patch
# would only land in the API server parent process).
#
# max-num-seqs=16 satisfies the FlashInfer kernel_warmup buffer requirement
# (vLLM warmup uses num_reqs=9, which exceeds max_num_seqs<9).
# --enforce-eager bypasses vLLM's cudagraph fast-path in fast_plan_decode,
# which itself has a second incompatibility (positional DLPack call with 15
# args into a 19-arg JIT function). Without enforce-eager every step would
# trip "Mismatched number of arguments when calling plan".
MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-16384}"
STARTING_VENV_DIR="${INFERENCE_BENCH_STARTING_VENV_DIR:-}"

if [ -n "${STARTING_VENV_DIR}" ] && [ -x "${STARTING_VENV_DIR}/bin/python" ]; then
    export VIRTUAL_ENV="${STARTING_VENV_DIR}"
    export PATH="${STARTING_VENV_DIR}/bin:${PATH}"
fi

export HF_HOME="${HF_HOME:-${HF_HOME_NEW:-${HOME}/hf_cache}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY_USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
# Prepend launcher dir so sitecustomize.py + compat_patch.py are auto-loaded
# by every Python process (including vLLM's spawned engine workers).
export PYTHONPATH="${LAUNCHER_DIR}:/home/agent/task/.local/lib/python3.10/site-packages:${PY_USER_SITE:-}:${PYTHONPATH:-}"

export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER_PREFILL=0

echo "=== vLLM Sc A arm1-patched: FlashInfer attention + FP8 weights + decode-plan shim ==="
echo "MODEL_ID=${MODEL_ID} HOST=${HOST} PORT=${PORT}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "LAUNCHER_DIR=${LAUNCHER_DIR}"
echo "PYTHONPATH=${PYTHONPATH}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 10240 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --kv-cache-dtype auto \
    --quantization fp8 \
    --enforce-eager \
    --trust-remote-code \
    --disable-log-stats
