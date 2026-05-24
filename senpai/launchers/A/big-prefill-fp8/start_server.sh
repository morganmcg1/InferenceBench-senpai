#!/usr/bin/env bash
set -euo pipefail

# Scenario A — one-shot long prefill + FP8 weights.
#
# Hypothesis (PR #78): Scenario A is prefill-bound. At concurrency=1 the
# 8192-token prompt dominates request latency, so chunked prefill costs more
# than it earns. Bumping --max-num-batched-tokens above the whole prompt and
# disabling chunked scheduling lets the prefill fire in one scheduler step.
# FP8 weight quantization makes the GEMM kernels faster on Blackwell-class
# GPUs while the MMLU-Pro quality gate is well-published as passing for
# Mistral-7B-Instruct-v0.3 under FP8.
#
# Keep this launcher derived from src/starting_points/vllm_running/start_server.sh
# so the final supervised relaunch shares the same env-handling preamble.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
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

echo "=== vLLM Inference Server (Scenario A: big-prefill-fp8) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=16384 (one-shot 8192-prompt prefill)"
echo "MAX_NUM_BATCHED_TOKENS=16384 (chunked prefill disabled)"
echo "QUANTIZATION=fp8 (weights only; KV cache stays default dtype)"
echo "HF_HOME=${HF_HOME}"
echo "==========================================================="

# Force integer device index — some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 16384 \
    --no-enable-chunked-prefill \
    --quantization fp8 \
    --trust-remote-code \
    --disable-log-stats
