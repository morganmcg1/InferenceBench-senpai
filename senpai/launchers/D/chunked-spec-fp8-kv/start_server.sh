#!/usr/bin/env bash
set -euo pipefail

# Scenario D launcher: balanced 4096->2048 burst-4 workload.
#
# Combines:
#   * --enable-chunked-prefill with --max-num-batched-tokens 8192 so an
#     arriving long prefill interleaves with in-flight decode tokens for the
#     other concurrent sequences instead of blocking them.
#   * n-gram speculative decoding (num_speculative_tokens=3, small enough to
#     keep accept rate high under 4-way concurrent decode).
#   * CUDA graphs (no --enforce-eager).
#   * FP8 KV cache to keep headroom for 4 concurrent 4k+2k contexts at
#     gpu-memory-utilization 0.92.
#   * --max-num-seqs 16 covers burst-4 with plenty of slack while keeping
#     decode batches modest.
#
# Foreground exec contract per InferenceBench task launcher spec.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-131072}"
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

# Default to FlashAttention; on RTX PRO 6000 (Blackwell) FlashInfer has been
# unreliable per advisor coordination notes.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

# Force integer device index — some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== Scenario D launcher: chunked prefill + n-gram spec (k=3) + FP8 KV + CUDA graphs ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "HF_HOME=${HF_HOME}"
echo "========================================================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --speculative-config '{"model":"[ngram]","num_speculative_tokens":3,"prompt_lookup_max":3,"prompt_lookup_min":2}' \
    --trust-remote-code \
    --disable-log-stats
