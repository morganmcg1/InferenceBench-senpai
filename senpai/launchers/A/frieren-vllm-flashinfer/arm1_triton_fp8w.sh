#!/usr/bin/env bash
set -euo pipefail

# Sc A arm1-Triton (PR #148 fallback): Triton attention backend + FP8 weights.
#
# Why this arm exists: the PR's primary arm1_flashinfer_fp8w.sh is blocked by a
# vLLM 0.11 / FlashInfer 0.6.11.post3 incompatibility. vLLM 0.11 pins
# flashinfer-python==0.3.1 and calls `fast_plan_decode` -> `decode_wrapper.plan`
# with positional args matching the 0.3.x signature. FlashInfer 0.6 inserted an
# extra `o_data_type` parameter, shifting `sm_scale` one slot. The vLLM forward
# pass then trips:
#   assert decode_wrapper._sm_scale == self.scale  # AssertionError
#
# Per PR #148's "Triton fallback" instruction, swap the attention backend to
# Triton (VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1). This bypasses both
# FlashAttention and FlashInfer wrappers and tests whether a Triton-tiled
# attention kernel beats FlashAttention 2 on Blackwell SM120.
#
# Everything else mirrors PR #137's FP8-weight winner. max-num-seqs=8 is fine
# here (the FlashInfer warmup buffer constraint only applies when
# VLLM_ATTENTION_BACKEND=FLASHINFER).
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

PY_USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
export PYTHONPATH="/home/agent/task/.local/lib/python3.10/site-packages:${PY_USER_SITE:-}:${PYTHONPATH:-}"

export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER_PREFILL=1

echo "=== vLLM Sc A arm1-Triton: Triton attention + FP8 weights ==="
echo "MODEL_ID=${MODEL_ID} HOST=${HOST} PORT=${PORT}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 10240 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --kv-cache-dtype auto \
    --quantization fp8 \
    --trust-remote-code \
    --disable-log-stats
