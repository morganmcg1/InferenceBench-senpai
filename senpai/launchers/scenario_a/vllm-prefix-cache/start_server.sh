#!/usr/bin/env bash
set -euo pipefail

# Scenario A round-2 follow-up to merged PR #83 (vllm-fastprefill, 1.246x).
# Single-lever delta: --enable-prefix-caching ON. All other flags match the
# merged Scenario A launcher exactly, so the comparison is apples-to-apples.
#
# Hypothesis: LongBench-v2 8k-prefill requests may share an instruction prefix.
# Prefix caching skips recomputing that shared prefix and should cut TTFT.

if [ -z "${PROBLEM_DIR:-}" ] || [ ! -f "${PROBLEM_DIR}/senpai/runtime_env.sh" ]; then
    for _cand in /workspace/senpai-frieren/target /workspace/senpai/target; do
        if [ -f "${_cand}/senpai/runtime_env.sh" ]; then
            PROBLEM_DIR="${_cand}"
            break
        fi
    done
fi
# shellcheck disable=SC1091
source "${PROBLEM_DIR}/senpai/runtime_env.sh"

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"

# Pin FLASH_ATTN explicitly. runtime_env.sh already disabled the implicit
# FlashInfer prefill path on this Blackwell card.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Force integer device index — some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== vLLM scenario-A prefix-cache-on ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "=================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 16384 \
    --no-enable-chunked-prefill \
    --enable-prefix-caching \
    --block-size 16 \
    --trust-remote-code \
    --disable-log-stats
