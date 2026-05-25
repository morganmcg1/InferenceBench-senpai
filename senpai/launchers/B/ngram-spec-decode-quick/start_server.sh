#!/usr/bin/env bash
set -euo pipefail

# Scenario B n-gram speculative decoding (quick probe, PR #117)
# Hypothesis: With concurrency=1 + output_len=8192 + ignore_eos=True, decode is
# memory-bandwidth-limited per token. n-gram speculative decoding drafts N
# tokens by lookup against the prompt + already-generated prefix and verifies
# them in one parallel forward pass — multiplying effective decode throughput
# when the model's output is repetitive (likely under ignore_eos heading past
# natural stops). Built on PR #114 Arm-1 BF16 base: CUDA graphs on, FLASH_ATTN,
# block-size 16, no prefix caching, no chunked prefill. BF16 weights to
# isolate the speculative-decode signal from the FP8 compounding.

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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

echo "=== vLLM Server: Scenario B n-gram speculative decoding (BF16 base) ==="
echo "MODEL_ID=${MODEL_ID} HOST=${HOST} PORT=${PORT} MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "Backend=FLASH_ATTN block-size=16 gpu-mem=0.92 max-num-seqs=32 batched=4096"
echo "Speculative: ngram k=4 prompt_lookup_min=2 max=4"
echo "============================================================================="

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 4096 \
    --block-size 16 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
    --trust-remote-code \
    --disable-log-stats
