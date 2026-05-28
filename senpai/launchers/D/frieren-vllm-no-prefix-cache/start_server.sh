#!/usr/bin/env bash
set -euo pipefail

# SENPAI scen-d-frieren Scenario D vLLM launcher — prefix-cache ablation.
# - chunked prefill ON, prefix caching OFF (ablation vs PR #155 balanced recipe),
#   wider batch, high GPU memory util, CUDA graphs ON (no --enforce-eager),
#   BF16 KV (auto), FLASH_ATTN backend.
# - LongBench-v2 prompts are mostly unique at c=1, so the prefix cache likely
#   has near-zero hit rate. This launcher tests whether removing it changes
#   the quick speedup meaningfully.
# - Targets the official task-local contract: foreground exec, OpenAI API.

if [ -f "${PROBLEM_DIR:-/workspace/senpai-scen-d-frieren/target}/senpai/runtime_env.sh" ]; then
    source "${PROBLEM_DIR:-/workspace/senpai-scen-d-frieren/target}/senpai/runtime_env.sh"
fi

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

PY_USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
export PYTHONPATH="/home/agent/task/.local/lib/python3.10/site-packages:${PY_USER_SITE:-}:${PYTHONPATH:-}"

echo "=== vLLM Inference Server (scen-d-frieren Scenario D vllm-no-prefix-cache) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "HF_HOME=${HF_HOME}"
echo "Levers: chunked-prefill ON, prefix-caching OFF, max-num-seqs=64,"
echo "        max-num-batched-tokens=16384, block-size=16, kv-cache-dtype=auto,"
echo "        gpu-memory-utilization=0.92, CUDA graphs ON, FLASH_ATTN backend."
echo "================================================================="

# Force integer device index — some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --block-size 16 \
    --kv-cache-dtype auto \
    --trust-remote-code \
    --disable-log-stats
