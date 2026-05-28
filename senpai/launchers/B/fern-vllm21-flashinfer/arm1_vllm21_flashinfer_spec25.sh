#!/usr/bin/env bash
# Sc B PR #190 arm1: PR #179 winner (FP8 + ngram spec25/lookup12) on vLLM 0.21.0 + FlashInfer.
# Hypothesis: PR #186 proved vLLM 0.21 + FlashInfer wins on Sc A (+2.9% TTFT).
# Sc B is decode-heavy (8192-out, conc=1) — FlashInfer decode kernel may improve
# TPOT. n-gram spec25 adds verify-step compute; FlashInfer may handle that batch
# pattern differently than FlashAttention.
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

# Add the venv's nvidia/*/include and nvidia/*/lib paths to CPATH and
# LD_LIBRARY_PATH so flashinfer JIT compilation at startup can find curand.h
# (and other CUDA 13 headers). The parent shell's runtime_env.sh discovery
# ran against the *system* python's site-packages, which does not include
# CUDA 13 headers from this per-PR venv.
_vllm21_site="${VLLM21_VENV}/lib/python3.10/site-packages/nvidia"
if [ -d "${_vllm21_site}" ]; then
    _venv_includes=""
    _venv_libs=""
    for _inc in "${_vllm21_site}"/*/include; do
        [ -d "${_inc}" ] && _venv_includes="${_venv_includes:+${_venv_includes}:}${_inc}"
    done
    for _lib in "${_vllm21_site}"/*/lib; do
        [ -d "${_lib}" ] && _venv_libs="${_venv_libs:+${_venv_libs}:}${_lib}"
    done
    if [ -n "${_venv_includes}" ]; then
        export CPATH="${_venv_includes}${CPATH:+:${CPATH}}"
        export C_INCLUDE_PATH="${_venv_includes}${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
        export CPLUS_INCLUDE_PATH="${_venv_includes}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
    fi
    if [ -n "${_venv_libs}" ]; then
        export LD_LIBRARY_PATH="${_venv_libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
fi
unset _vllm21_site _venv_includes _venv_libs _inc _lib

export HF_HOME="${HF_HOME:-${HOME}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

# FlashInfer prefill enabled (match PR #186 arm2 exactly).
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER_PREFILL=0

echo "=== Sc B PR #190 arm1: vLLM 0.21 + FlashInfer + FP8 + spec25/12 ==="
echo "vLLM version: $(${VLLM21_VENV}/bin/python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'unknown')"

exec "${VLLM21_VENV}/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 2048 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --kv-cache-dtype auto \
    --quantization fp8 \
    --attention-backend FLASHINFER \
    --speculative-config '{"method":"ngram","num_speculative_tokens":25,"prompt_lookup_max":12,"prompt_lookup_min":2}' \
    --trust-remote-code \
    --disable-log-stats
