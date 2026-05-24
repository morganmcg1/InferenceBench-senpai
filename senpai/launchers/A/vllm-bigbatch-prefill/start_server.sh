#!/usr/bin/env bash
set -euo pipefail

# vLLM bigbatch-prefill recipe for InferenceBench Scenario A.
#
# Scenario A is input-heavy (8192 in / 1024 out, burst concurrency 1, primary
# metric `1/ttft.p50`). The dominant cost is the long prefill matmul/attention
# before the first generated token. This launcher:
#   * Sets --max-num-batched-tokens 16384 so the full 8k prefill runs in a
#     single forward pass (no chunked-prefill scheduling overhead, no split
#     across steps).
#   * Disables chunked prefill and prefix caching (burst, no shared prefixes).
#   * Uses --max-num-seqs 1 to match the burst-concurrency-1 workload.
#   * Uses --kv-cache-dtype fp8 to free memory headroom (marginal TTFT cost).
#   * Keeps CUDA graphs enabled (no --enforce-eager).
#   * Reduces --max-model-len to 16384 to speed model load and shrink block
#     tables (8192 in + 1024 out fits well under that).

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

echo "=== vLLM Inference Server (Scenario A: bigbatch-prefill) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "max-num-seqs=1  max-num-batched-tokens=16384  kv-cache-dtype=fp8"
echo "no-chunked-prefill  no-prefix-caching  block-size=16  gpu-mem-util=0.92"
echo "HF_HOME=${HF_HOME}"
echo "============================================================"

# Force integer device index — some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Pod-local: flashinfer JIT top-k/top-p sampler fails to compile on this image
# (CUDA-13 nvcc vs CUDA-12 pip headers). Native torch sampler is exact at
# concurrency 1. Confirmed by r3-frieren on PR #37.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 16384 \
    --kv-cache-dtype fp8 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --block-size 16 \
    --trust-remote-code \
    --disable-log-stats
