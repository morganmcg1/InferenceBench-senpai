#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-16384}"
STARTING_VENV_DIR="${INFERENCE_BENCH_STARTING_VENV_DIR:-}"

if [ -n "${STARTING_VENV_DIR}" ] && [ -x "${STARTING_VENV_DIR}/bin/python" ]; then
    export VIRTUAL_ENV="${STARTING_VENV_DIR}"
    export PATH="${STARTING_VENV_DIR}/bin:${PATH}"
fi

if [ -n "${PROBLEM_DIR:-}" ] && [ -f "$PROBLEM_DIR/senpai/runtime_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$PROBLEM_DIR/senpai/runtime_env.sh"
fi

export HF_HOME="${HF_HOME:-${HF_HOME_NEW:-${HOME}/hf_cache}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

PY_USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
export PYTHONPATH="/home/agent/task/.local/lib/python3.10/site-packages:${PY_USER_SITE:-}:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Blackwell sm_120 (RTX PRO 6000): FlashInfer JIT does not build for sm_120 and
# FA3 fp8 KV is Hopper-only. Default to TRITON_ATTN and keep vLLM off the
# flashinfer path entirely (sampler + prefill). Per advisor 2026-05-24 07:42 UTC.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_DISABLE_FLASHINFER_PREFILL="${VLLM_DISABLE_FLASHINFER_PREFILL:-1}"

echo "=== vLLM Scenario A launcher: bf16 W + FP8 KV + TRITON_ATTN + no chunked prefill ==="
echo "MODEL_ID=${MODEL_ID} HOST=${HOST} PORT=${PORT} MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "ATTN_BACKEND=${VLLM_ATTENTION_BACKEND} FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER} DISABLE_FLASHINFER_PREFILL=${VLLM_DISABLE_FLASHINFER_PREFILL}"

# FP8 W8A8 dropped per advisor 2026-05-24 08:03 UTC: r4-frieren confirmed MMLU-Pro
# observed 0.282 vs baseline 0.298 (ratio 0.946 < tau 0.95) on Mistral-7B with
# --quantization fp8. Quality is a model-level property. FP8 KV cache retained.
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 16384 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --disable-log-stats
