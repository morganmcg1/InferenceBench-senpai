#!/usr/bin/env bash
set -euo pipefail

# Scenario A (input-heavy, 8K prefill / 1K decode, concurrency 1, burst).
# Tuned for single-stream prefill latency on RTX PRO 6000 Blackwell.
#
# Levers in this launcher:
#   * FlashAttention backend (stable on RTX PRO 6000; FlashInfer prefill stays
#     disabled per senpai/runtime_env.sh defaults).
#   * --max-num-batched-tokens 16384 so an 8K prefill processes in a single
#     forward pass instead of being split.
#   * --enable-chunked-prefill ON for Arm 1; the evaluator's burst profile at
#     concurrency 1 may still benefit from chunked scheduling cost amortization.
#   * --max-model-len 10240 to bound the KV cache pool (8K prefill + 1K decode
#     fits with room to spare and no oversize allocation).
#   * BF16 KV cache (no --kv-cache-dtype fp8 — FlashAttn rejects FP8 KV here).
#   * CUDA graphs ON (default; no --enforce-eager).
#   * Prefix caching OFF (LongBench prompts have no shared prefix).
#   * --seed 248 to match the dataset/quality seed used by the scoring assets.
#
# Foreground exec; binds to ${HOST:-0.0.0.0}:${PORT:-8000}; no &/nohup/setsid.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Backend pin: keep FlashAttention; runtime_env.sh keeps FlashInfer prefill off.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

# Force integer device index — some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== vLLM Scenario A prefill tune (Arm 1: chunked prefill, 16384 batched tokens) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "===================================================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --seed 248 \
    --tokenizer-mode auto \
    --trust-remote-code \
    --max-model-len 10240 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 16 \
    --block-size 16 \
    --no-enable-prefix-caching \
    --disable-log-stats
