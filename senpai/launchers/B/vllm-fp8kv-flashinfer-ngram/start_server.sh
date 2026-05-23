#!/usr/bin/env bash
set -euo pipefail

# Scenario B (output-heavy / TPOT) vLLM launcher.
# Combines three orthogonal levers expected to compound on long-decode workloads:
#   - FP8 KV cache to halve per-token KV bandwidth during decode
#   - FlashInfer attention backend, decode-optimised vs default FLASH_ATTN
#   - n-gram speculative decoding (5 draft tokens, lookup [1, 5])
# Keep the final server process in the foreground; the benchmark harness
# supervises launch and teardown outside this script.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"
STARTING_VENV_DIR="${INFERENCE_BENCH_STARTING_VENV_DIR:-}"

if [ -n "${STARTING_VENV_DIR}" ] && [ -x "${STARTING_VENV_DIR}/bin/python" ]; then
    export VIRTUAL_ENV="${STARTING_VENV_DIR}"
    export PATH="${STARTING_VENV_DIR}/bin:${PATH}"
fi

# HF cache resolution: explicit env wins; otherwise prefer pod-shared /tmp
# senpai-cache (already populated by other students in this run) before
# falling back to the per-student default.
if [ -z "${HF_HOME:-}" ]; then
    if [ -n "${HF_HOME_NEW:-}" ] && [ -d "${HF_HOME_NEW}" ]; then
        HF_HOME="${HF_HOME_NEW}"
    elif [ -d "${XDG_CACHE_HOME:-/tmp/senpai-cache}/huggingface/hub" ]; then
        HF_HOME="${XDG_CACHE_HOME:-/tmp/senpai-cache}/huggingface"
    elif [ -d "/tmp/senpai-cache/huggingface/hub" ]; then
        HF_HOME="/tmp/senpai-cache/huggingface"
    else
        HF_HOME="${HOME}/hf_cache"
    fi
fi
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

PY_USER_SITE="$(python3 -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
export PYTHONPATH="/home/agent/task/.local/lib/python3.10/site-packages:${PY_USER_SITE:-}:${PYTHONPATH:-}"

# FlashInfer decode backend — specialised for decode-dominant workloads.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"

# Pod-local JIT caches to avoid cross-pod contention on shared dirs.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/triton_r4fern_scB}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${TMPDIR:-/tmp}/inductor_r4fern_scB}"

# Force integer device index — some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== vLLM Sc-B launcher: FP8-KV + FlashInfer + n-gram speculative ==="
echo "MODEL_ID=${MODEL_ID}  HOST=${HOST}  PORT=${PORT}  MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "HF_HOME=${HF_HOME}  HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "===================================================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":5,"prompt_lookup_min":1}' \
    --trust-remote-code \
    --disable-log-stats
