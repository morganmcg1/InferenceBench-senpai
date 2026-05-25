#!/usr/bin/env bash
set -euo pipefail

# Fallback launcher for InferenceBench Scenario D when the SGLang
# tanjiro-tuned-sglang-balanced recipe cannot boot on this pod.
#
# Container kernel skew: sglang 0.5.12.post1 requires sgl_kernel symbols
# (gptq_gemm, moe_sum, rotary_embedding, fused_experts, etc.) that the
# installed sgl_kernel 0.3.4.post1 does not export. SGLang therefore fails
# at the ServerArgs init stage before ever touching the GPU.
#
# This launcher mirrors src/starting_points/vllm_running/start_server.sh with
# two PR-mandated tweaks: bump gpu-memory-utilization to 0.95 and clamp
# max-model-len to 8192 so the KV-cache budget matches the Scenario D
# input+output token shape (4096+2048).

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-8192}"

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

echo "=== vLLM Inference Server (tanjiro-fallback-vllm-default-d) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "================================================"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --disable-log-stats
