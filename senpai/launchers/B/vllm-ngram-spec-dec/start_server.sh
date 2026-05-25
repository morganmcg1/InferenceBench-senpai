#!/usr/bin/env bash
set -euo pipefail

# Scenario B (output-heavy, 1K prefill / 8K decode, concurrency 1, burst).
# Primary target: tpot.p50 via n-gram speculative decoding.
#
# Levers in this launcher:
#   * FlashAttention backend (stable on RTX PRO 6000; FlashInfer prefill stays
#     disabled per senpai/runtime_env.sh defaults).
#   * --speculative-config with method=ngram, propose 5 tokens, lookup 2-4 grams.
#     vLLM verifies 5+1 positions in a single forward pass.
#   * --max-num-seqs 2: n-gram spec dec internally needs 2 sequence slots
#     (draft proposal + verification) at concurrency 1.
#   * --max-num-batched-tokens 2048: each decode step verifies up to 6 tokens;
#     2048 is comfortably above this and avoids prefill chunking interactions.
#   * --block-size 32: larger KV blocks reduce management overhead at 8K decode.
#   * --no-enable-chunked-prefill: concurrency 1 sees no benefit; keep prefill
#     monolithic.
#   * --max-model-len 10240 bounds KV cache pool (1K in + 8K out fits).
#   * BF16 KV cache (no fp8); CUDA graphs ON; prefix caching OFF; seed 248.
#
# Foreground exec; binds to ${HOST:-0.0.0.0}:${PORT:-8000}; no &/nohup/setsid.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== vLLM Scenario B — n-gram speculative decoding ==="
echo "MODEL_ID=${MODEL_ID}  HOST=${HOST}  PORT=${PORT}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "======================================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --seed 248 \
    --tokenizer-mode auto \
    --trust-remote-code \
    --max-model-len 10240 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 2048 \
    --block-size 32 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
    --disable-log-stats
