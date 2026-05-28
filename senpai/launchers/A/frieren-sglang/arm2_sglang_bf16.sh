#!/usr/bin/env bash
# Sc A arm2 (PR #183): SGLang + BF16 weights (engine-swap control).
# Identical to arm1 EXCEPT --quantization fp8 is removed. This isolates
# the engine-swap contribution from FP8 weights at Sc A's prefill-dominant
# conc=1 regime. If arm2 is competitive with arm1, the engine swap (not FP8)
# is the driver; if arm2 underperforms, FP8 weights are doing the work.
set -euo pipefail

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-16384}"
SGLANG_VENV="${SGLANG_VENV:-/tmp/inferencebench-engine-venvs/sglang-pr-183}"

if [ ! -x "${SGLANG_VENV}/bin/python" ]; then
    echo "[bootstrap] SGLang venv not found at ${SGLANG_VENV}; creating now..." >&2
    mkdir -p "$(dirname "${SGLANG_VENV}")"
    if command -v uv >/dev/null 2>&1; then
        uv venv "${SGLANG_VENV}" --python 3.10 --seed
        PIP_REQUIRE_VIRTUALENV=false "${SGLANG_VENV}/bin/pip" install --upgrade pip wheel setuptools
        PIP_REQUIRE_VIRTUALENV=false "${SGLANG_VENV}/bin/pip" install "sglang[all]"
    else
        python3 -m venv "${SGLANG_VENV}"
        PIP_REQUIRE_VIRTUALENV=false "${SGLANG_VENV}/bin/pip" install --upgrade pip wheel setuptools
        PIP_REQUIRE_VIRTUALENV=false "${SGLANG_VENV}/bin/pip" install "sglang[all]"
    fi
    echo "[bootstrap] SGLang venv ready at ${SGLANG_VENV}." >&2
fi
export VIRTUAL_ENV="${SGLANG_VENV}"
export PATH="${SGLANG_VENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_CANDIDATES=(
    "${SGLANG_LIB_DIR:-}"
    "${SCRIPT_DIR}"
    "${SCRIPT_DIR}/lib"
    "/workspace/senpai/target/senpai/launchers/A/frieren-sglang"
    "/workspace/senpai/target/senpai/launchers/C/fern-sglang-mem-push/lib"
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

echo "=== SGLang Inference Server (Sc A arm2 — BF16 control, conc=1, no-chunk) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "SGLANG_VENV=${SGLANG_VENV}"
echo "SGLANG_LIB_DIR=${SGLANG_LIB_DIR}"
echo "==========================================================================="

exec python3 -m sglang.launch_server \
    --model-path "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --context-length "${MAX_MODEL_LEN}" \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 16384 \
    --max-running-requests 1 \
    --disable-radix-cache \
    --attention-backend triton \
    --kv-cache-dtype auto \
    --trust-remote-code
