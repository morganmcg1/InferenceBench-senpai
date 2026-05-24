#!/usr/bin/env bash
set -euo pipefail

# vLLM launcher: FP8 weights + FP8 KV cache + n-gram speculative decoding.
# Optimized for Scenario B (output-heavy decode, 1024 in / 8192 out, concurrency 1)
# on RTX PRO 6000 Blackwell.

source "${PROBLEM_DIR}/senpai/runtime_env.sh"

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-${HF_HOME_NEW:-${HOME}/hf_cache}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

# FlashInfer JIT for sm_120f (Blackwell) needs CUDA 12 nvcc but the runtime
# ships CUDA 13 nvcc, which fails to compile against CUDA 12 wheel headers.
# Force PyTorch sampler + disable FlashInfer prefill so vLLM does not JIT.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER_PREFILL=1
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"

echo "=== vLLM FP8 + n-gram speculative decoding (Scenario B) ==="
echo "MODEL_ID=${MODEL_ID}  HOST=${HOST}  PORT=${PORT}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.92 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 4096 \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
    --trust-remote-code \
    --disable-log-stats
