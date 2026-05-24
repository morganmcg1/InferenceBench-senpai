#!/usr/bin/env bash
set -euo pipefail

# Scenario B (output-heavy, conc=1) launcher.
# vLLM with FP8 weights + FP8 KV cache + n-gram prompt-lookup speculative decoding.
# Pairs roughly halved per-step memory traffic with multi-token speculation to
# cut TPOT, which is the scoring objective for Scenario B's burst profile.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-131072}"
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

# Source SENPAI runtime helpers if PROBLEM_DIR is reachable. Best-effort so
# the launcher remains usable from a clean container that supplies its own
# CUDA paths.
_runtime_env_candidates=(
    "${PROBLEM_DIR:-}/senpai/runtime_env.sh"
    "/workspace/senpai-r1-fern/target/senpai/runtime_env.sh"
    "/home/agent/target/senpai/runtime_env.sh"
    "/target/senpai/runtime_env.sh"
)
for _cand in "${_runtime_env_candidates[@]}"; do
    if [ -n "${_cand}" ] && [ -f "${_cand}" ]; then
        # shellcheck disable=SC1090
        source "${_cand}"
        break
    fi
done
unset _runtime_env_candidates _cand

# Force integer device index — some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Use FlashAttention backend on Blackwell for the decode-heavy single-stream path.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

SPEC_CFG='{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'

echo "=== vLLM Scenario B: FP8 weights+KV + n-gram speculative decoding ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "HF_HOME=${HF_HOME}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "SPEC_CFG=${SPEC_CFG}"
echo "===================================================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.90 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --block-size 16 \
    --speculative-config "${SPEC_CFG}" \
    --trust-remote-code \
    --disable-log-stats
