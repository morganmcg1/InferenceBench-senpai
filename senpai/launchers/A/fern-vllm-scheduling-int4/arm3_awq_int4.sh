#!/usr/bin/env bash
# arm3: INT4 AWQ probe. Halves weight size vs FP8 (4x vs BF16),
# targeting prefill weight-load time at Sc A conc=1.
# Architecture preserved (MistralForCausalLM, hidden=4096, vocab=32768);
# quality gate must pass at n>=16 to promote.
set -euo pipefail
# Hardcode AWQ checkpoint; alias via --served-model-name so the evaluator
# (which addresses INFERENCE_BENCH_BASE_MODEL) routes to this server.
MODEL_ID="solidrust/Mistral-7B-Instruct-v0.3-AWQ"
SERVED_NAME="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
HOST="${HOST:-0.0.0.0}"; PORT="${PORT:-8000}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-16384}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Prefer awq_marlin (faster on SM8x+); fallback to plain awq is handled by
# arm3b_awq_plain.sh if marlin kernels are unavailable on SM120.
exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" --served-model-name "${SERVED_NAME}" \
  --host "${HOST}" --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --kv-cache-dtype auto \
  --quantization awq_marlin \
  --trust-remote-code --disable-log-stats
