#!/usr/bin/env bash
set -euo pipefail

# Scenario B (output-heavy, burst concurrency 1, 1024 in -> 8192 out):
# decode-bound. Levers:
#   - n-gram speculative decoding (prompt_lookup, no draft model) for TPOT.
#   - FP8 KV cache (default) to free VRAM; set IB_KV_CACHE_DTYPE=auto to fall
#     back to bf16 KV (FlashAttention path) if FlashInfer's FP8 prefill kernel
#     cannot JIT-compile against the system CUDA toolkit.
#   - CUDA graphs ON (no --enforce-eager).
#   - chunked prefill / prefix caching OFF (burst-concurrency-1 sequential
#     long decode; prefix caching helps repeated prompts, which scenario B
#     does not have).

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

PY_USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
export PYTHONPATH="/home/agent/task/.local/lib/python3.10/site-packages:${PY_USER_SITE:-}:${PYTHONPATH:-}"

# Force an integer device index — vLLM cannot parse GPU UUID values some pods set.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# KV cache dtype: FlashInfer (only backend on Blackwell that accepts FP8 KV)
# needs to JIT-compile its FP8 prefill kernels, which fails when the pod has
# CUDA 13 nvcc but FlashInfer's bundled headers are CUDA 12. Default to FP8 +
# FlashInfer; fall back to bf16 + FlashAttention if the env requests it.
KV_CACHE_DTYPE="${IB_KV_CACHE_DTYPE:-auto}"
if [ "${KV_CACHE_DTYPE}" = "fp8" ]; then
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
else
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
fi

# FlashInfer's top-k/top-p sampler op JIT-compiles against system nvcc.
# The pod ships CUDA 13 nvcc while FlashInfer 0.6.11 bundles CUDA 12 headers,
# so the JIT build fails. Force PyTorch-native sampling instead — this is a
# pure-CUDA op, just slightly less throughput on the sampler.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

echo "=== vLLM Scenario B launcher (n-gram spec + CUDA graphs) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
echo "============================================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
    --trust-remote-code \
    --disable-log-stats
