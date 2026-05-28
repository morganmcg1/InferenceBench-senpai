#!/usr/bin/env bash
# Sc C PR #188 arm2: PR #181 winning config + --attention-backend fa3.
# Hypothesis: FlashAttention 3 may execute faster than Triton on Blackwell SM120
# for the long-context (16384) high-conc (256) Sc C workload. Orthogonal probe
# to arm1's FlashInfer. All other knobs match PR #181.
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
LIB_REL_PATH="senpai/launchers/C/fern-sglang-flashinfer/lib"
LIB_CANDIDATES=(
    "${SGLANG_LIB_DIR:-}"
    "${SCRIPT_DIR}/lib"
    "${SCRIPT_DIR}/${LIB_REL_PATH}"
    "${PROBLEM_DIR:-}/${LIB_REL_PATH}"
    "/workspace/senpai/target/${LIB_REL_PATH}"
    "/workspace/senpai-fern/target/${LIB_REL_PATH}"
    "/workspace/senpai/target/senpai/launchers/C/fern-sglang-mem-push/lib"
    "/workspace/senpai/target/senpai/launchers/C/fern-sglang-fp8wt/lib"
    "/workspace/senpai/target/senpai/launchers/C/frieren-sglang-fp8kv/lib"
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
    exit 1
fi
export LD_LIBRARY_PATH="${SGLANG_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "=== SGLang Inference Server (Sc C PR #188 arm2 — PR #181 + FA3 backend) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "ATTENTION_BACKEND=fa3 (probe vs PR #181 triton)"
echo "===================================================="

exec python3 -m sglang.launch_server \
    --model-path "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --context-length "${MAX_MODEL_LEN}" \
    --mem-fraction-static 0.90 \
    --chunked-prefill-size 8192 \
    --schedule-policy lpm \
    --max-running-requests 256 \
    --attention-backend fa3 \
    --kv-cache-dtype fp8_e5m2 \
    --quantization fp8 \
    --trust-remote-code
