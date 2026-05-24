#!/usr/bin/env bash
set -euo pipefail

# Scenario A (prefill-bound, 8192 in / 1024 out, burst c=1).
# Goal: minimize TTFT.p50 via FP8 weights + dense KV + chunked prefill OFF on
# vLLM v1, with CUDA graphs ON. The original PR hypothesis was FP8 weights +
# FP8 KV + FlashInfer prefill, but on this RTX PRO 6000 + CUDA 13.2 + vLLM
# 0.11 image neither FlashInfer (FlashInferImpl forward() asserts under FP8
# KV) nor FLASH_ATTN (NotImplementedError: FlashAttention does not support fp8
# kv-cache on this device) accept FP8 KV. So this launcher defaults to
# attention=FLASH_ATTN and kv-cache-dtype=auto, keeping every other lever from
# the hypothesis. Override with SENPAI_FORCE_FLASHINFER=1 or
# SENPAI_KV_CACHE_DTYPE=fp8 to re-attempt the original combo when the toolchain
# is fixed.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# Mistral-7B-Instruct-v0.3 config caps positional embeddings at 32768; scenario A
# only needs 9216 (8192 prefill + 1024 decode). 16384 leaves headroom without
# triggering VLLM_ALLOW_LONG_MAX_MODEL_LEN RoPE-NaN risk.
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-16384}"
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

# Force integer device index — some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Attention backend selection. FlashInfer is the goal; FLASH_ATTN is the
# documented fallback (see PR #63). The PR text allows FLASH_ATTN if FLASHINFER
# fails — on this CUDA 13.2 + sm_120f image FlashInfer JIT compiles and imports
# but vLLM 0.11's FlashInferImpl asserts in forward() with FP8 KV cache, so we
# default to FLASH_ATTN. Set SENPAI_FORCE_FLASHINFER=1 to force-try FlashInfer.
ATTENTION_BACKEND="FLASH_ATTN"
if [ "${SENPAI_FORCE_FLASHINFER:-0}" = "1" ]; then
    if python3 -c "import importlib, sys; importlib.import_module('flashinfer'); sys.exit(0)" 2>/dev/null; then
        ATTENTION_BACKEND="FLASHINFER"
    else
        echo "[start_server] flashinfer import failed; staying on FLASH_ATTN" >&2
    fi
fi
export VLLM_ATTENTION_BACKEND="${ATTENTION_BACKEND}"

# Keep FlashInfer sampling off — its JIT path is fragile on Blackwell/CUDA 13.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# senpai/runtime_env.sh prepends the pip nvidia/cuda_runtime/include (CUDA 12.8)
# to CPATH for vLLM extensions. FlashInfer's JIT then mixes those CUDA 12.8
# headers with the CUDA 13.2 nvcc toolchain — vector_types.h is missing the
# longlong4_32a type that libcudacxx 13 references. Unset CPATH so nvcc resolves
# cooperative_groups.h / vector_types.h from /usr/local/cuda/include (CUDA 13.2).
unset CPATH
# Documented escape hatch for libcudacxx vs CUDA toolkit version mismatch.
export FLASHINFER_EXTRA_CUDAFLAGS="${FLASHINFER_EXTRA_CUDAFLAGS:--DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK}"

echo "=== vLLM Inference Server (FP8 + FlashInfer prefill, scenario A) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "HF_HOME=${HF_HOME}"
echo "===================================================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs 32 \
    --max-num-batched-tokens 16384 \
    --no-enable-chunked-prefill \
    --enable-prefix-caching \
    --quantization fp8 \
    --kv-cache-dtype "${SENPAI_KV_CACHE_DTYPE:-auto}" \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --disable-log-stats
