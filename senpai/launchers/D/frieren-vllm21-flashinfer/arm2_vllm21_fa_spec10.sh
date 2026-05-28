#!/usr/bin/env bash
# Sc D PR #189 arm2: PR #152 winner config on vLLM 0.21.0 + FlashAttention (control).
# This isolates the engine-version effect from the attention-backend effect.
# arm1 = vLLM 0.21 + FlashInfer (hypothesis), arm2 = vLLM 0.21 + FA (control).
# If arm2 beats PR #152 (vLLM 0.11 + FA): engine-version contributed.
# If arm1 > arm2: FlashInfer contributed on top of engine upgrade.
set -euo pipefail

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-16384}"
VLLM21_VENV="${VLLM21_VENV:-/tmp/inferencebench-engine-venvs/vllm12-pr-186}"

if [ ! -x "${VLLM21_VENV}/bin/python" ]; then
    echo "[bootstrap] vLLM 0.21 venv not found at ${VLLM21_VENV}; creating now..." >&2
    mkdir -p "$(dirname "${VLLM21_VENV}")"
    if command -v uv >/dev/null 2>&1; then
        uv venv "${VLLM21_VENV}" --python 3.10 --seed
        PIP_REQUIRE_VIRTUALENV=false "${VLLM21_VENV}/bin/pip" install --upgrade pip wheel setuptools
        PIP_REQUIRE_VIRTUALENV=false "${VLLM21_VENV}/bin/pip" install "vllm>=0.12.0"
    elif python3 -c "import ensurepip" >/dev/null 2>&1; then
        python3 -m venv "${VLLM21_VENV}"
        PIP_REQUIRE_VIRTUALENV=false "${VLLM21_VENV}/bin/pip" install --upgrade pip wheel setuptools
        PIP_REQUIRE_VIRTUALENV=false "${VLLM21_VENV}/bin/pip" install "vllm>=0.12.0"
    else
        echo "FATAL: cannot bootstrap vLLM venv." >&2
        exit 1
    fi
    echo "[bootstrap] vLLM 0.21 venv ready at ${VLLM21_VENV}." >&2
fi
export VIRTUAL_ENV="${VLLM21_VENV}"
export PATH="${VLLM21_VENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== Sc D PR #189 arm2: vLLM 0.21.0 + FlashAttention + spec10/6 (control) ==="
echo "vLLM version: $(${VLLM21_VENV}/bin/python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'unknown')"

exec "${VLLM21_VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 4096 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --kv-cache-dtype auto \
    --quantization fp8 \
    --speculative-config '{"method":"ngram","num_speculative_tokens":10,"prompt_lookup_max":6,"prompt_lookup_min":2}' \
    --trust-remote-code \
    --disable-log-stats
