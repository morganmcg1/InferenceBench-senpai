#!/usr/bin/env bash
set -euo pipefail

# vLLM launcher recipe for Scenario D (balanced general workload) on
# RTX PRO 6000 Blackwell. Combines the levers confirmed by PR #39
# Scenario C (FP8 weights + FP8 KV cache + FLASH_ATTN + chunked prefill)
# with light n-gram speculative decoding tuned for 2K outputs and a
# concurrency=4 burst.
#
# Foreground exec, no nohup / no &, follows the InferenceBench server contract.

PROBLEM_DIR_CANDIDATE="${PROBLEM_DIR:-}"
if [ -z "${PROBLEM_DIR_CANDIDATE}" ]; then
    if [ -f "/workspace/senpai-r1-frieren/target/senpai/runtime_env.sh" ]; then
        PROBLEM_DIR_CANDIDATE="/workspace/senpai-r1-frieren/target"
    elif [ -f "$(dirname "${BASH_SOURCE[0]}")/../../../runtime_env.sh" ]; then
        PROBLEM_DIR_CANDIDATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
    fi
fi

if [ -n "${PROBLEM_DIR_CANDIDATE}" ] && [ -f "${PROBLEM_DIR_CANDIDATE}/senpai/runtime_env.sh" ]; then
    # shellcheck disable=SC1091
    source "${PROBLEM_DIR_CANDIDATE}/senpai/runtime_env.sh"
    # On this pod vLLM/torch link against the system CUDA 13 toolkit. The pip
    # nvidia-* (cu12) packages that runtime_env.sh wires into CPATH conflict
    # with the system CUDA 13 CCCL headers and break FlashInfer JIT compile
    # (CCCL toolkit-compatibility check). Drop the full CPATH but restore a
    # minimal curand include path, since /usr/local/cuda-13.2 ships without
    # the cuRAND headers FlashInfer needs at JIT time.
    unset CPATH
    if [ -d "/usr/local/lib/python3.10/dist-packages/nvidia/curand/include" ]; then
        export CPATH="/usr/local/lib/python3.10/dist-packages/nvidia/curand/include"
    fi
fi

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"
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

# Blackwell-safe attention backend. FlashInfer JIT compile is flaky on this
# clone due to CUDA-13 / cu12 header mismatch; FLASH_ATTN is the proven path
# from PR #39. vLLM 0.21+ also accepts --attention-backend, but the legacy env
# var keeps us compatible with both vLLM 0.21 and earlier builds.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

# Force integer device index — some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== vLLM Inference Server (Scenario D, FP8 weights + FP8 KV + ngram spec) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "HF_HOME=${HF_HOME}"
echo "===================================================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.90 \
    --block-size 16 \
    --speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":3,"prompt_lookup_min":2}' \
    --trust-remote-code \
    --disable-log-stats
