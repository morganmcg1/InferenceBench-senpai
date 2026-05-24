#!/usr/bin/env bash
set -euo pipefail

# Some pods inject PROBLEM_DIR=target/ via the launch harness; that relative
# stub breaks the source below. Probe known absolute roots when the current
# value does not point at a real runtime_env.sh.
if [ -z "${PROBLEM_DIR:-}" ] || [ ! -f "${PROBLEM_DIR}/senpai/runtime_env.sh" ]; then
    for _candidate in \
        /workspace/senpai-fern/target \
        /workspace/senpai/target \
        ; do
        if [ -f "${_candidate}/senpai/runtime_env.sh" ]; then
            PROBLEM_DIR="${_candidate}"
            break
        fi
    done
fi
# shellcheck disable=SC1091
source "$PROBLEM_DIR/senpai/runtime_env.sh"

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"

# Pin FLASH_ATTN. runtime_env.sh already disabled implicit FlashInfer.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Force integer device index — some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== vLLM scenario-B n-gram speculative decoding ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "speculative: ngram, num_speculative_tokens=5"
echo "===================================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 4096 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --block-size 16 \
    --speculative-model "[ngram]" \
    --num-speculative-tokens 5 \
    --trust-remote-code \
    --disable-log-stats
