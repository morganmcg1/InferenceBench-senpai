#!/usr/bin/env bash
set -euo pipefail

# Scenario C — high-concurrency tight-context FP8 launcher.
#
# Hypothesis (PR #97): Scenario C (1024-in / 1024-out, concurrency=128) is
# throughput-bound, not prefill- or decode-latency-bound. The H100 reference
# already shows vLLM-default ~48.69x on Sc. C — i.e. built-in continuous
# batching saturates the scheduler. Headroom on RTX PRO 6000 Blackwell is
# step compute density, not scheduling.
#
# Design choices:
#   * --max-model-len 2048: minimum that fits (1024 prompt + 1024 output).
#     For Mistral-7B-Instruct-v0.3 (GQA, num_kv_heads=8, head_dim=128, 32 layers,
#     fp16), KV per token is 128 KiB. At 128 seqs * 2048 tokens that is ~32 GiB
#     of KV cache, leaving comfortable headroom for FP8 weights + activations
#     on the 96 GiB device.
#   * --max-num-seqs 128: exactly the Sc. C concurrency target; the server
#     never has to queue requests on its own side.
#   * --max-num-batched-tokens 16384: large enough to absorb the prefill burst
#     from many simultaneous arrivals (chunked prefill handles any overflow).
#   * --enable-chunked-prefill: at concurrency=128 requests are constantly
#     arriving and finishing; chunked prefill keeps decode running while new
#     prefills come in.
#   * --quantization fp8: confirmed safe on this hardware by PRs #78 (Sc. A
#     MMLU-Pro ratio 1.000) and #82 (Sc. D ratio 0.9933). FP8 cuts SwiGLU+QKV
#     GEMM latency without quality loss at tau=0.95.
#
# Keep this launcher derived from src/starting_points/vllm_running/start_server.sh
# so the final supervised relaunch shares the same env-handling preamble.

MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
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

echo "=== vLLM Inference Server (Scenario C: highload-fp8) ==="
echo "MODEL_ID=${MODEL_ID}"
echo "HOST=${HOST} PORT=${PORT}"
echo "MAX_MODEL_LEN=2048 (tight: 1024 prompt + 1024 output)"
echo "MAX_NUM_SEQS=128 (matches Sc. C concurrency)"
echo "MAX_NUM_BATCHED_TOKENS=16384 (absorb prefill bursts)"
echo "CHUNKED_PREFILL=enabled"
echo "QUANTIZATION=fp8 (weights only)"
echo "HF_HOME=${HF_HOME}"
echo "==========================================================="

# Force integer device index — some nodes expose GPU UUIDs which vLLM cannot parse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --quantization fp8 \
    --trust-remote-code \
    --disable-log-stats
