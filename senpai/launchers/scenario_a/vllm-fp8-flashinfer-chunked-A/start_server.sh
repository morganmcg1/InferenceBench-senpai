#!/usr/bin/env bash
set -euo pipefail

# Scenario A (input-heavy, TTFT) launcher.
# Levers: FP8 weights + FP8 KV cache + FlashAttention backend +
# chunked prefill with max_num_batched_tokens == one full 8192-token prefill.
# Goal: beat the TPE 2h vLLM reference of 4.48x speedup over PyTorch on TTFT.
#
# Backend note: an earlier FlashInfer attempt failed during CUDA graph
# capture on this Blackwell GPU (sm_120f). Fell back to FLASH_ATTN, which
# is what the PR fallback list calls out as the alternate backend. We also
# leave a margin on gpu-memory-utilization (0.90) and use the supported
# --attention-backend CLI flag introduced in vLLM 0.21.

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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Disable vLLM usage stats; attention backend is set via the CLI flag below.
export VLLM_NO_USAGE_STATS="1"

PY_USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
export PYTHONPATH="/home/agent/task/.local/lib/python3.10/site-packages:${PY_USER_SITE:-}:${PYTHONPATH:-}"

echo "=== vLLM Scenario A — FP8 + FlashInfer + chunked prefill ==="
echo "MODEL_ID=${MODEL_ID}  HOST=${HOST}  PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "============================================================"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.90 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 8192 \
    --block-size 16 \
    --enable-chunked-prefill \
    --attention-backend FLASH_ATTN \
    --trust-remote-code \
    --disable-log-stats
