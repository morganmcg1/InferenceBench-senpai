#!/usr/bin/env bash
# Scenario C launcher: vLLM with FP8 KV cache + larger batches.
#
# Hypothesis: halving per-token KV memory (FP8 vs FP16) lets vLLM admit ~1.5-2x
# more concurrent sequences before paging, increasing decode-phase throughput on
# Scenario C. KV-only quantization preserves model weights, so the MMLU-Pro
# quality gate should still pass.
#
# Foreground server (exec). Respects HOST, PORT, INFERENCE_BENCH_BASE_MODEL,
# INFERENCE_BENCH_MAX_MODEL_LEN.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate the SENPAI runtime env. When this script is copied into a task
# workspace at task/start_server.sh, the SENPAI repo is no longer adjacent —
# fall back to PROBLEM_DIR / a small search.
RUNTIME_ENV_CANDIDATES=(
    "${PROBLEM_DIR:-}/senpai/runtime_env.sh"
    "${SCRIPT_DIR}/../../../runtime_env.sh"
    "${SCRIPT_DIR}/../../../../senpai/runtime_env.sh"
    "/workspace/senpai-r5-tanjiro/target/senpai/runtime_env.sh"
)
for candidate in "${RUNTIME_ENV_CANDIDATES[@]}"; do
    if [ -n "${candidate}" ] && [ -f "${candidate}" ]; then
        # shellcheck disable=SC1090
        source "${candidate}"
        break
    fi
done

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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# FlashInfer is needed on some Blackwell builds for FP8 KV cache. If JIT
# compilation fails to start the server, override with
# VLLM_ATTENTION_BACKEND=FLASH_ATTN (or unset) and relaunch — document the
# fallback in the PR comment.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"

echo "=== vLLM Inference Server (Scenario C, FP8 KV cache) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "HF_HOME=${HF_HOME}"
echo "========================================================"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs 512 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.93 \
    --no-enable-prefix-caching \
    --enable-chunked-prefill \
    --kv-cache-dtype fp8 \
    --quantization none \
    --tokenizer-mode auto \
    --trust-remote-code \
    --disable-log-stats
