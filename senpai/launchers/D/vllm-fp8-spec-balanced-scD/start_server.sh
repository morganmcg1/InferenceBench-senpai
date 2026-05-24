#!/usr/bin/env bash
set -euo pipefail

# Scenario D (balanced, 4096 in / 2048 out, burst c=4).
# Geomean of 1/ttft.p50, 1/tpot.p50, req/s.
#
# RTX PRO 6000 / Blackwell compatibility notes from advisor (2026-05-24 16:29/16:34):
#  - FlashInfer attention backend fails JIT on this hardware/toolkit; do not export
#    VLLM_ATTENTION_BACKEND=FLASHINFER. runtime_env.sh already disables FlashInfer
#    prefill and sampler.
#  - FlashAttention rejects --kv-cache-dtype fp8 on RTX PRO 6000; use auto/default.
#  - --quantization fp8 is unproven to boot here; omit it for the safe arm and
#    rely on the systems-level levers below for the speedup.
#
# Active levers (all RTX PRO 6000 compatible):
#   * n-gram prompt-lookup speculative decoding (free tokens on the long-output
#     side -> TPOT and req/s).
#   * chunked prefill at max_num_batched_tokens=8192, max_num_seqs=64 so decode
#     steps interleave with prefill chunks instead of starving at c=4 -> TTFT
#     and total throughput.
#   * CUDA graphs (no --enforce-eager) -> decode latency.
#   * --enable-prefix-caching -> repeat-prompt acceleration if present.
#   * gpu-memory-utilization 0.90 -> larger KV budget for batching.

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

# Force integer device index — some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Keep FlashInfer disabled on RTX PRO 6000 per advisor. runtime_env.sh sets
# VLLM_DISABLE_FLASHINFER_PREFILL=1 and VLLM_USE_FLASHINFER_SAMPLER=0. We do
# not override VLLM_ATTENTION_BACKEND here; vLLM will pick FLASH_ATTN.

echo "=== vLLM Inference Server (scenario D balanced, n-gram spec + chunked prefill) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-unset}"
echo "VLLM_DISABLE_FLASHINFER_PREFILL=${VLLM_DISABLE_FLASHINFER_PREFILL:-unset}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-default}"
echo "HF_HOME=${HF_HOME}"
echo "======================================================================"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":5,"prompt_lookup_min":2}' \
    --trust-remote-code
