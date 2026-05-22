#!/usr/bin/env bash
set -euo pipefail

# v2 of senpai/launchers/scenario_b/vllm-fp8-ngram-spec-B/start_server.sh.
# Addresses the vLLM scheduler warning from the v1 full eval:
#   "max_num_scheduled_tokens is set to 4096 based on the speculative decoding
#    settings. This may lead to suboptimal performance. Consider increasing
#    max_num_batched_tokens to accommodate the additional draft token slots."
#
# Changes vs v1:
#   --max-num-batched-tokens 4096 -> 8192   (room for 5 draft tokens x slots)
#   --max-num-seqs            32  -> 48     (more concurrent slots for 64-burst)
# Everything else (FP8 weights, FP8 KV, ngram spec, block-size, gpu-mem-util)
# is held constant so the comparison vs v1 is clean.

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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# FlashInfer is faster than FlashAttention for decode-heavy (small batch, large KV)
export VLLM_ATTENTION_BACKEND="FLASHINFER"
export VLLM_NO_USAGE_STATS="1"

PY_USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
export PYTHONPATH="/home/agent/task/.local/lib/python3.10/site-packages:${PY_USER_SITE:-}:${PYTHONPATH:-}"

# Add pip-installed CUDA shared libs (cudart, cublas, ...) to LD_LIBRARY_PATH.
# vLLM/torch wheels bring their own CUDA 12 runtime, which can be missing from
# the system ld cache on containers built against a different CUDA toolkit.
NVIDIA_PIP_LIBS="$(python3 -c 'import glob,os; print(":".join(sorted({os.path.join(d,"lib") for d in glob.glob("/usr/local/lib/python3.10/dist-packages/nvidia/*") if os.path.isdir(os.path.join(d,"lib"))})))' 2>/dev/null || true)"
if [ -n "${NVIDIA_PIP_LIBS}" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_PIP_LIBS}:${LD_LIBRARY_PATH:-}"
fi

# Do NOT add nvidia pip include dirs to CPATH. Doing so pulls in
# /usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/include/cooperative_groups.h,
# which then transitively includes /usr/local/cuda/include/cccl/.../cuda_toolkit.h,
# tripping a toolkit/version compatibility assert during FlashInfer FP8 attention
# JIT. The system CUDA 13.2 toolkit at /usr/local/cuda/include already ships
# cooperative_groups.h and curand.h, so nvcc finds them on its default path.

echo "=== vLLM Scenario B — FP8 + N-gram speculative decoding (v2) ==="
echo "MODEL_ID=${MODEL_ID}  HOST=${HOST}  PORT=${PORT}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.95 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 48 \
    --max-num-batched-tokens 8192 \
    --block-size 16 \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
    --trust-remote-code \
    --disable-log-stats
