#!/usr/bin/env bash
set -euo pipefail

# Scenario B (output-heavy, burst concurrency 1, 1024 in / 8192 out)
# TPOT-focused vLLM launcher.
#
# Levers:
#   * --kv-cache-dtype fp8         halves KV-cache bandwidth per decode step
#   * --max-num-seqs 1             matches burst concurrency 1
#   * --max-num-batched-tokens 2048 keeps prefill chunk small/cheap
#   * --no-enable-chunked-prefill  avoid scheduler tick cost during 8k decode
#   * --no-enable-prefix-caching   no shared prefix in burst single-stream
#   * --block-size 16              fine-grained KV blocks
#   * --max-model-len 16384        1024 in + 8192 out fits comfortably
#
# CUDA graphs remain ENABLED (no --enforce-eager) to eliminate Python overhead
# in the decode loop. Full-weight FP8 quantization (--quantization fp8) is
# explicitly out of scope here.

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

echo "=== vLLM Scenario B FP8 KV TPOT-burst1 launcher ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "HF_HOME=${HF_HOME}"
echo "===================================================="

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Flashinfer's JIT-compiled top-k/top-p sampler kernel fails to build on this
# pod (CUDA 13 nvcc vs CUDA 12 headers shipped by pip nvidia-* packages).
# Fall back to the native torch sampler — it's accurate and adds negligible
# latency at concurrency 1, which is what scenario B targets anyway.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 2048 \
    --kv-cache-dtype fp8 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --block-size 16 \
    --trust-remote-code \
    --disable-log-stats
