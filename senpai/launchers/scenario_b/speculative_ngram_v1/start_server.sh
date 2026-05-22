#!/usr/bin/env bash
set -euo pipefail

# SENPAI Scenario B candidate: vLLM 0.11 n-gram speculative decoding tuned for
# the long-decode workload (1024 input / 8192 output, burst concurrency 1).
#
# Hypothesis (see r1-fern/sc-b-speculative): TPOT-dominated decoding benefits
# from n-gram speculation because LongBench-derived prompts repeat substrings
# in the generated continuation. Combine with chunked prefill for prefill
# headroom and FLASH_ATTN as the attention backend. KV cache dtype is selected
# at the bottom of this script via the SENPAI_KV_CACHE_DTYPE env var
# (auto | fp8) so the same launcher recipe covers the B1 and B2 arms.
#
# Arm selection knobs (set via env when copying to ./start_server.sh):
#   SENPAI_NUM_SPECULATIVE_TOKENS   default 3 (B1, B2)
#   SENPAI_PROMPT_LOOKUP_MAX        default 4
#   SENPAI_PROMPT_LOOKUP_MIN        default 2
#   SENPAI_KV_CACHE_DTYPE           default auto (B1, B3); fp8 for B2
#
# The final process is exec-ed in the foreground per the InferenceBench server
# contract; HOST and PORT are honored.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"
STARTING_VENV_DIR="${INFERENCE_BENCH_STARTING_VENV_DIR:-}"

NUM_SPEC="${SENPAI_NUM_SPECULATIVE_TOKENS:-3}"
LOOKUP_MAX="${SENPAI_PROMPT_LOOKUP_MAX:-4}"
LOOKUP_MIN="${SENPAI_PROMPT_LOOKUP_MIN:-2}"
KV_CACHE_DTYPE="${SENPAI_KV_CACHE_DTYPE:-auto}"

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

# Attention backend (vLLM 0.21 accepts both env var and CLI flag; we pass both).
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

# Disable flashinfer top-k/top-p sampler. On Blackwell (sm_120) the flashinfer
# JIT pipeline needs curand.h which is not present in this CUDA install, so the
# sampler fails to compile at warmup. The pytorch-native sampler handles
# temperature=0.7 sampling correctly for scenario B's quality+speed workload.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# Force integer device index — some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SPEC_JSON=$(python3 -c "import json; print(json.dumps({'method':'ngram','num_speculative_tokens':int('${NUM_SPEC}'),'prompt_lookup_max':int('${LOOKUP_MAX}'),'prompt_lookup_min':int('${LOOKUP_MIN}')}, separators=(',', ':')))")

echo "=== vLLM Inference Server (Scenario B speculative ngram v1) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER}"
echo "NUM_SPECULATIVE_TOKENS=${NUM_SPEC}"
echo "PROMPT_LOOKUP_MAX=${LOOKUP_MAX} PROMPT_LOOKUP_MIN=${LOOKUP_MIN}"
echo "KV_CACHE_DTYPE=${KV_CACHE_DTYPE}"
echo "HF_HOME=${HF_HOME}"
echo "speculative-config JSON=${SPEC_JSON}"
echo "================================================================"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 16 \
    --enable-chunked-prefill \
    --attention-backend "${VLLM_ATTENTION_BACKEND}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --speculative-config "${SPEC_JSON}" \
    --trust-remote-code \
    --disable-log-stats
