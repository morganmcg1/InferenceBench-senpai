#!/usr/bin/env bash
# Sc C arm3 (PR #167): PR #144 winner (BF16 KV + LPM + radix) + FP8 weights only.
# This is the diagnostic CONTROL arm: it isolates the FP8-weight contribution
# without the FP8 KV layer. Compare vs PR #144 (24.305x, BF16 weights+KV) and
# PR #151 (27.497x, BF16 weights + FP8 KV).
# If this arm regresses below PR #144 → SGLang FP8 weights HURT Sc C and the
# composition in arm1/arm2 is risky. If it wins → FP8 weights help and the
# composition should win further.
set -euo pipefail

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-16384}"
SGLANG_VENV="${SGLANG_VENV:-/tmp/inferencebench-engine-venvs/sglang-pr-167}"

if [ ! -x "${SGLANG_VENV}/bin/python" ]; then
    echo "[bootstrap] SGLang venv not found at ${SGLANG_VENV}; creating now..." >&2
    mkdir -p "$(dirname "${SGLANG_VENV}")"
    if command -v uv >/dev/null 2>&1; then
        uv venv "${SGLANG_VENV}" --python 3.10 --seed
        PIP_REQUIRE_VIRTUALENV=false "${SGLANG_VENV}/bin/pip" install --upgrade pip wheel setuptools
        PIP_REQUIRE_VIRTUALENV=false "${SGLANG_VENV}/bin/pip" install "sglang[all]"
    elif python3 -c "import ensurepip" >/dev/null 2>&1; then
        python3 -m venv "${SGLANG_VENV}"
        PIP_REQUIRE_VIRTUALENV=false "${SGLANG_VENV}/bin/pip" install --upgrade pip wheel setuptools
        PIP_REQUIRE_VIRTUALENV=false "${SGLANG_VENV}/bin/pip" install "sglang[all]"
    else
        echo "FATAL: cannot bootstrap SGLang venv: neither 'uv' nor python3 ensurepip is available." >&2
        exit 1
    fi
    echo "[bootstrap] SGLang venv ready at ${SGLANG_VENV}." >&2
fi
export VIRTUAL_ENV="${SGLANG_VENV}"
export PATH="${SGLANG_VENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_REL_PATH="senpai/launchers/C/fern-sglang-fp8wt/lib"
LIB_CANDIDATES=(
    "${SGLANG_LIB_DIR:-}"
    "${SCRIPT_DIR}/lib"
    "${SCRIPT_DIR}/${LIB_REL_PATH}"
    "${PROBLEM_DIR:-}/${LIB_REL_PATH}"
    "/workspace/senpai/target/${LIB_REL_PATH}"
    "/workspace/senpai-fern/target/${LIB_REL_PATH}"
    "/workspace/senpai/target/senpai/launchers/C/frieren-sglang-fp8kv/lib"
    "/workspace/senpai/target/senpai/launchers/C/fern-sglang-sc-c/lib"
)
SGLANG_LIB_DIR=""
for candidate in "${LIB_CANDIDATES[@]}"; do
    if [ -n "${candidate}" ] && [ -f "${candidate}/libnuma.so.1" ]; then
        SGLANG_LIB_DIR="${candidate}"
        break
    fi
done
if [ -z "${SGLANG_LIB_DIR}" ]; then
    echo "FATAL: bundled libnuma.so.1 not found." >&2
    for candidate in "${LIB_CANDIDATES[@]}"; do
        [ -n "${candidate}" ] && echo "  - ${candidate}/libnuma.so.1" >&2
    done
    exit 1
fi
export LD_LIBRARY_PATH="${SGLANG_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "=== SGLang Inference Server (Sc C arm3 — FP8 weights ONLY, BF16 KV @ mem 0.85) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "SGLANG_VENV=${SGLANG_VENV}"
echo "SGLANG_LIB_DIR=${SGLANG_LIB_DIR}"
echo "===================================================="

exec python3 -m sglang.launch_server \
    --model-path "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --context-length "${MAX_MODEL_LEN}" \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 8192 \
    --schedule-policy lpm \
    --max-running-requests 256 \
    --attention-backend triton \
    --quantization fp8 \
    --trust-remote-code
