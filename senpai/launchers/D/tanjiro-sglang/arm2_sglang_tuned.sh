#!/usr/bin/env bash
# SGLang tuned arm for Scenario D (PR #138):
#   --mem-fraction-static 0.85
#   --chunked-prefill-size 4096
#   --schedule-policy lpm
#   --max-running-requests 64
#
# Relaunch-safety contract:
#   * The launcher does NOT call apt-get or otherwise mutate the host.
#   * `libnuma.so.1` (required by SGLang's sgl_kernel SM100 binary on Ubuntu
#     base images that ship without libnuma) is shipped in this PR under
#     `senpai/launchers/D/tanjiro-sglang/lib/`. The launcher locates that dir
#     even when copied to a task workspace's `start_server.sh`.
#   * The launcher requires the SGLang venv. Recreate it from a clean
#     container with:
#         python "$PROBLEM_DIR/senpai/create_engine_venv.py" --engine sglang --pr 138
#     or, when ensurepip is missing, with uv:
#         uv venv /tmp/inferencebench-engine-venvs/sglang-pr-138 --python 3.10 --seed
#         /tmp/inferencebench-engine-venvs/sglang-pr-138/bin/pip install "sglang[all]"
set -euo pipefail

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-16384}"
SGLANG_VENV="${SGLANG_VENV:-/tmp/inferencebench-engine-venvs/sglang-pr-138}"

if [ ! -x "${SGLANG_VENV}/bin/python" ]; then
    echo "FATAL: SGLang venv not found at ${SGLANG_VENV}." >&2
    echo "Recreate with: uv venv ${SGLANG_VENV} --python 3.10 --seed && ${SGLANG_VENV}/bin/pip install 'sglang[all]'" >&2
    exit 1
fi
export VIRTUAL_ENV="${SGLANG_VENV}"
export PATH="${SGLANG_VENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_REL_PATH="senpai/launchers/D/tanjiro-sglang/lib"
LIB_CANDIDATES=(
    "${SGLANG_LIB_DIR:-}"
    "${SCRIPT_DIR}/lib"
    "${SCRIPT_DIR}/${LIB_REL_PATH}"
    "${PROBLEM_DIR:-}/${LIB_REL_PATH}"
    "/workspace/senpai/target/${LIB_REL_PATH}"
)
SGLANG_LIB_DIR=""
for candidate in "${LIB_CANDIDATES[@]}"; do
    if [ -n "${candidate}" ] && [ -f "${candidate}/libnuma.so.1" ]; then
        SGLANG_LIB_DIR="${candidate}"
        break
    fi
done
if [ -z "${SGLANG_LIB_DIR}" ]; then
    echo "FATAL: bundled libnuma.so.1 not found. Searched:" >&2
    for candidate in "${LIB_CANDIDATES[@]}"; do
        [ -n "${candidate}" ] && echo "  - ${candidate}/libnuma.so.1" >&2
    done
    echo "Hint: set SGLANG_LIB_DIR to the absolute path of the bundled lib dir." >&2
    exit 1
fi
export LD_LIBRARY_PATH="${SGLANG_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "=== SGLang Inference Server (arm2 tuned) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "SGLANG_VENV=${SGLANG_VENV}"
echo "SGLANG_LIB_DIR=${SGLANG_LIB_DIR}"
echo "============================================"

exec python3 -m sglang.launch_server \
    --model-path "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --context-length "${MAX_MODEL_LEN}" \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 4096 \
    --schedule-policy lpm \
    --max-running-requests 64 \
    --attention-backend triton \
    --trust-remote-code
