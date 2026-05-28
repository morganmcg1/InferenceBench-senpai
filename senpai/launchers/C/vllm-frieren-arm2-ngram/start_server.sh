#!/usr/bin/env bash
set -euo pipefail

# Scenario C vLLM high-concurrency launcher (scen-c-frieren, Arm E).
# Arm A baseline + n-gram speculative decoding.
# Rationale: Scenario C has output_len=1024 with ignore_eos=true (long decode).
# N-gram spec decode finds repeated token spans in the prompt and proposes
# them; under long decode and structured prompts this can amortize TPOT.
# Chunked prefill stays ON (V1 default; user flag is no-op anyway).

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"
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

# Keep FlashInfer prefill disabled by default on RTX PRO 6000 shakedown.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_DISABLE_FLASHINFER_PREFILL="${VLLM_DISABLE_FLASHINFER_PREFILL:-1}"

echo "=== vLLM Scenario C (frieren arm E, n-gram spec decode) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "HF_HOME=${HF_HOME}"
echo "========================================"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 384 \
    --max-num-batched-tokens 16384 \
    --kv-cache-dtype auto \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":5,"prompt_lookup_min":3}' \
    --trust-remote-code \
    --disable-log-stats
