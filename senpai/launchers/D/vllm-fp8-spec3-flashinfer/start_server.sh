#!/usr/bin/env bash
set -euo pipefail

# Scenario D (balanced general workload) vLLM launcher.
#
# Strategy: FP8 weights + FP8 KV cache (shrinks decode + KV footprint at c=4),
# FlashInfer attention, light 3-token n-gram speculative decoding, and chunked
# prefill so four overlapping requests share prefill budget evenly with decode.
# Prefix caching is disabled because Scenario D draws random LongBench-v2
# prompts that do not share prefixes.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"
STARTING_VENV_DIR="${INFERENCE_BENCH_STARTING_VENV_DIR:-}"

if [ -n "${STARTING_VENV_DIR}" ] && [ -x "${STARTING_VENV_DIR}/bin/python" ]; then
    export VIRTUAL_ENV="${STARTING_VENV_DIR}"
    export PATH="${STARTING_VENV_DIR}/bin:${PATH}"
fi

# Defensive: if the active Python has nvidia/* dev headers installed via pip
# (common when CUDA system headers are incomplete), expose them to nvcc so
# FlashInfer's JIT compile (curand.h, cuda_runtime.h, ...) can find them.
PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [ -n "${PYTHON_BIN}" ]; then
    NVIDIA_INC_DIRS="$("${PYTHON_BIN}" - <<'PY' 2>/dev/null || true
import os, glob, site
roots = []
for fn in (site.getsitepackages, site.getusersitepackages):
    try:
        v = fn()
        roots += v if isinstance(v, list) else [v]
    except Exception:
        pass
seen = set()
out = []
for root in roots:
    for inc in glob.glob(os.path.join(root, "nvidia", "*", "include")):
        if inc not in seen:
            seen.add(inc)
            out.append(inc)
print(":".join(out))
PY
)"
    if [ -n "${NVIDIA_INC_DIRS}" ]; then
        export CPATH="${NVIDIA_INC_DIRS}${CPATH:+:${CPATH}}"
        export CPLUS_INCLUDE_PATH="${NVIDIA_INC_DIRS}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
    fi
fi

export HF_HOME="${HF_HOME:-${HF_HOME_NEW:-${HOME}/hf_cache}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

PY_USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
export PYTHONPATH="/home/agent/task/.local/lib/python3.10/site-packages:${PY_USER_SITE:-}:${PYTHONPATH:-}"

# Some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Attention backend selection.
#
# On sm_120 (RTX PRO 6000 Blackwell), vLLM 0.11.0's FlashInfer backend hits the
# `decode_wrapper._sm_scale == self.scale` assertion at warmup whenever FP8 KV
# cache is enabled (with OR without n-gram spec decoding). The advisor's
# documented escalation for that scenario is to fall back to the FLASH_ATTN
# backend while keeping FP8 weights + FP8 KV cache + chunked prefill, and
# submit that as the terminal arm.
#
# Set VLLM_ATTENTION_BACKEND=FLASHINFER explicitly via env to re-enable the
# original recipe if/when the upstream bug is fixed.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_NO_USAGE_STATS=1

echo "=== vLLM Scenario-D balanced launcher (FP8 + FP8-KV + FlashInfer + 3-token n-gram spec dec) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "HF_HOME=${HF_HOME}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "================================================================================"

# Failure-handling notes (Scenario D, sm_120 RTX PRO 6000 Blackwell, vLLM 0.11.0):
#   1. 3-token n-gram speculative decoding tripped the FlashInfer decode-wrapper
#      `_sm_scale` assertion at warmup; per the PR's failure handling we drop
#      --speculative-config first.
#   2. The same assertion fires with FP8 weights + FP8 KV cache + FlashInfer +
#      chunked prefill even WITHOUT spec dec. The TRTLLM decode path (which
#      sidesteps the assertion) is not supported on sm_120 (it needs SM 100),
#      so forcing VLLM_USE_TRTLLM_ATTENTION=1 cannot rescue FlashInfer here.
#   3. Per the advisor's escalation path, fall back to the FLASH_ATTN backend
#      and keep FP8 weights + FP8 KV cache + chunked prefill. Submit that as
#      the terminal arm for Scenario D.
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --block-size 16 \
    --disable-log-stats \
    --trust-remote-code
