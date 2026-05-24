#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# Mistral-7B-Instruct-v0.3 max_position_embeddings=32768; all A-D scenario
# token budgets fit under this cap and vLLM 0.11 rejects 131072 without
# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1.
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"
STARTING_VENV_DIR="${INFERENCE_BENCH_STARTING_VENV_DIR:-}"

if [ -n "${STARTING_VENV_DIR}" ] && [ -x "${STARTING_VENV_DIR}/bin/python" ]; then
    export VIRTUAL_ENV="${STARTING_VENV_DIR}"
    export PATH="${STARTING_VENV_DIR}/bin:${PATH}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-${HF_HOME_NEW:-${HOME}/hf_cache}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

# Disable FlashInfer top-k/top-p sampler; the JIT-compile path tripped CCCL
# version-check conflicts between the python wheel CUDA 12.8 headers and the
# on-pod CUDA 13.2 nvcc. The PyTorch-native sampler is slightly slower per
# step but does not need ninja+nvcc at boot.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

PY_USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
export PYTHONPATH="/home/agent/task/.local/lib/python3.10/site-packages:${PY_USER_SITE:-}:${PYTHONPATH:-}"

if [ -n "${PROBLEM_DIR:-}" ] && [ -r "${PROBLEM_DIR}/senpai/runtime_env.sh" ]; then
    # shellcheck disable=SC1091
    source "${PROBLEM_DIR}/senpai/runtime_env.sh"
fi

# runtime_env.sh prepends every nvidia/<lib>/include path to CPATH. That
# includes nvidia/cuda_runtime/include, which carries CUDA 12.8 cooperative_
# groups and other headers that conflict with the on-pod CUDA 13.2 nvcc when
# FlashInfer JIT-compiles. The CCCL version check fires with
# "CUDA compiler and CUDA toolkit headers are incompatible".
# Fix: drop everything from CPATH, then re-add ONLY the wheel paths nvcc
# actually needs (curand.h is required by flashinfer sampling kernels and
# does not live in /usr/local/cuda/include).
unset CPATH
for d in /usr/local/lib/python3.10/dist-packages/nvidia/curand/include; do
    if [ -d "$d" ]; then
        export CPATH="$d${CPATH:+:$CPATH}"
    fi
done

echo "=== vLLM Inference Server (Scenario B ngram-speculative, default KV) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "HF_HOME=${HF_HOME}"
echo "===================================================================="

# Advisor URGENT note 16:29 UTC: FlashInfer JIT + FP8 KV are failed paths on
# this RTX PRO 6000 / CUDA 13.2 pod. The PR instructions allow dropping
# --kv-cache-dtype fp8 as a fallback when FP8 KV + speculative decoding is
# unsupported. Keep n-gram speculative decoding (the primary hypothesis) and
# let vLLM pick the default KV dtype that boots cleanly.
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.93 \
    --trust-remote-code \
    --disable-log-stats \
    --max-num-seqs 32 \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'
