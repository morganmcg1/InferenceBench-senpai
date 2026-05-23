#!/usr/bin/env bash
set -euo pipefail

# Scenario B (output-heavy, concurrency 1, target 8192 output tokens):
#   * keep per-decode-step latency small (CUDA graphs enabled by default)
#   * minimize KV-cache bytes for the long decode (fp8 KV)
#   * use FlashInfer decode kernels (typically faster than FA at concurrency 1)
#   * cap max-num-seqs since true concurrency is 1 in the burst profile
#   * chunked prefill on for the short 1k input phase
#   * prefix caching off (unique LongBench-v2 prompts)

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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# On this Blackwell pod (sm_120f) the originally planned attention +
# kv-cache combos do not work end-to-end:
#   * VLLM_ATTENTION_BACKEND=FLASHINFER fails because the FlashInfer JIT
#     can't compile its sampling kernel ("CUDA compiler and CUDA toolkit
#     headers are incompatible").
#   * VLLM_ATTENTION_BACKEND=FLASH_ATTN refuses fp8 kv-cache on this
#     device ("FlashAttention does not support fp8 kv-cache on this
#     device").
# So we land with FLASH_ATTN + auto kv-cache dtype (fp16). The output-
# heavy wins still come from CUDA graphs (default on), chunked prefill
# for the short input phase, prefix-caching off (unique LongBench-v2
# prompts), block_size=16, and max-num-seqs=8 — all geared toward keeping
# per-decode-step latency small at concurrency 1.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

echo "=== Scenario B vLLM launcher (auto kv-dtype + flash-attn + cuda graphs) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "HF_HOME=${HF_HOME}"
echo "==================================================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 2048 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --block-size 16 \
    --trust-remote-code \
    --disable-log-stats
