#!/usr/bin/env bash
set -euo pipefail

PROBLEM_DIR="${PROBLEM_DIR:-/workspace/senpai/target}"
# Pod env may set PROBLEM_DIR to a relative path like "target/" that does not
# resolve from /tmp/ib-D/task; fall back to known absolute SENPAI roots.
if [ ! -f "${PROBLEM_DIR}/senpai/runtime_env.sh" ]; then
  for _cand in /workspace/senpai-tanjiro/target /workspace/senpai/target; do
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
# Arm1: 8192 (one 4k prefill chunk + decode). Arm2: 16384 (absorb full 4-prompt
# burst prefill 4*4096 in one batch). Default to arm1 (the committed winner).
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"

# Pin FLASH_ATTN. runtime_env.sh already disabled implicit FlashInfer.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Force integer device index — some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== vLLM scenario-D balanced ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "================================"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --block-size 16 \
    --trust-remote-code \
    --disable-log-stats
