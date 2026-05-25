#!/usr/bin/env bash
set -euo pipefail

# SGLang launcher for InferenceBench Scenario D (burst, c=4, 4k input + 2k output).
#
# Hypothesis: SGLang's longest-prefix-match (lpm) scheduler can exploit the
# shared chat-template prefix, while triton attention is confirmed to work on
# RTX PRO 6000 Blackwell.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SCHEDULE_POLICY="${SGLANG_SCHEDULE_POLICY:-lpm}"

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

# Disable the SGLang JIT DeepGEMM path: the deep_gemm package shipped in this
# container is older than what SGLang 0.5.12.post1's wrapper expects (the
# wrapper imports `deep_gemm.utils.layout` which does not exist), and any FP8
# grouped-matmul fast path is irrelevant for the BF16 Mistral-7B-Instruct
# workload this launcher targets.
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"

echo "=== SGLang Inference Server (tanjiro-tuned-sglang-balanced) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "SCHEDULE_POLICY=${SCHEDULE_POLICY}"
echo "SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM}"
echo "================================================================"

exec python3 -m sglang.launch_server \
    --model-path "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --context-length 8192 \
    --max-running-requests 64 \
    --mem-fraction-static 0.90 \
    --chunked-prefill-size 8192 \
    --max-prefill-tokens 16384 \
    --schedule-policy "${SCHEDULE_POLICY}" \
    --attention-backend triton \
    --trust-remote-code
