#!/usr/bin/env bash
set -euo pipefail

# Sc A arm1 (PR #148): FlashInfer attention backend + FP8 weight quantization.
# Hypothesis: FlashInfer's fused Tensor-Core-tiled prefill kernel beats
# FlashAttention 2 on the concurrency-1 8192-token Sc A prefill.
# VLLM_USE_FLASHINFER_SAMPLER=0 keeps native sampler (SM120-safe);
# VLLM_DISABLE_FLASHINFER_PREFILL=0 overrides runtime_env.sh shakedown default
# so the prefill path actually uses FlashInfer (the whole point of arm1).
#
# NOTE: max-num-seqs=16 (vs PR #137's 8). vLLM's FlashInfer kernel_warmup runs
# _dummy_run(num_tokens=16, create_mixed_batch=True) which requires
# num_reqs = 8 decode + 1 prefill = 9 slots in self.seq_lens.np; max_num_seqs<9
# triggers a ValueError during server startup. The FlashInfer kernel itself
# compiles cleanly on Blackwell SM120 -- this is only a vLLM warmup buffer
# constraint.
MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY_USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
export PYTHONPATH="/home/agent/task/.local/lib/python3.10/site-packages:${PY_USER_SITE:-}:${PYTHONPATH:-}"

export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER_PREFILL=0

echo "=== vLLM Sc A arm1: FlashInfer attention + FP8 weights ==="
echo "MODEL_ID=${MODEL_ID} HOST=${HOST} PORT=${PORT}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER}"
echo "VLLM_DISABLE_FLASHINFER_PREFILL=${VLLM_DISABLE_FLASHINFER_PREFILL}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 10240 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --kv-cache-dtype auto \
    --quantization fp8 \
    --trust-remote-code \
    --disable-log-stats
