#!/usr/bin/env bash
set -euo pipefail

# Tuned vLLM launcher for InferenceBench Scenario D (burst, c=4,
# 4096 input + 2048 output, balanced geomean(1/ttft.p50, 1/tpot.p50, req/s)).
#
# Started from the PR-defined vLLM fallback (gpu-memory-utilization 0.95,
# max-model-len 8192) and adds Scenario-D-specific tuning:
#
# - `--max-num-seqs 32` is comfortably above the c=4 burst so the scheduler
#   never bottlenecks, while keeping the seq table small enough for cheap
#   per-step bookkeeping.
# - `--max-num-batched-tokens 8192` fits a single 4096 prefill chunk plus
#   tail tokens comfortably under c=4 concurrency.
# - `--enable-chunked-prefill` matters at c=4: prefills compete with decode
#   tokens, and chunked prefill lets them interleave cleanly.
# - `--enable-prefix-caching` helps for the shared chat-template prefix that
#   Mistral-Instruct uses across requests (the user content itself is
#   unique per request, but the Mistral [INST]/[/INST] boilerplate is shared
#   and cheap to cache).
# - `--max-model-len 8192` matches the 4096+2048 token shape with margin and
#   reduces the per-sequence KV slot reservation vs the runtime_env default
#   of 32768. RTX PRO 6000 has ~96 GB VRAM so KV is not the bottleneck, but
#   the smaller table reduces sampler/scheduler overhead.
# - FlashAttention backend (vLLM auto-detect) with BF16 KV cache. The pod
#   has `VLLM_USE_FLASHINFER_SAMPLER=0` and `VLLM_DISABLE_FLASHINFER_PREFILL=1`
#   set by `senpai/runtime_env.sh`; we do not re-enable FlashInfer here.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${VLLM_TUNED_MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${VLLM_TUNED_MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${VLLM_TUNED_MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEM_UTIL="${VLLM_TUNED_GPU_MEM_UTIL:-0.95}"

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

echo "=== vLLM Inference Server (tanjiro-tuned-vllm-balanced) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
echo "GPU_MEM_UTIL=${GPU_MEM_UTIL}"
echo "VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-unset}"
echo "VLLM_DISABLE_FLASHINFER_PREFILL=${VLLM_DISABLE_FLASHINFER_PREFILL:-unset}"
echo "================================================"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --trust-remote-code \
    --disable-log-stats
