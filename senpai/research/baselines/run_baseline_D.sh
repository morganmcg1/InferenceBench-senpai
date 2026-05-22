#!/usr/bin/env bash
# Run the transformers PyTorch baseline for Scenario D on this Blackwell pod.
# Produces baseline_metrics.json that other PRs can use as --baseline-metrics-json.
#
# Time-budget-aware: defaults to --request-limit 24 because torch concurrency=1
# at 96 requests × 2048 tokens would exceed the 2h research window. 24 requests
# gives statistically reasonable p50 latency estimates while staying under
# ~15 min of sequential decode.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/senpai-r5-tanjiro/target}"
VENV="${VENV:-$REPO_ROOT/.vllm_venv}"
MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
PORT="${BASELINE_PORT:-9000}"
REQUEST_LIMIT="${BASELINE_REQUEST_LIMIT:-24}"
REQUEST_TIMEOUT_S="${BASELINE_REQUEST_TIMEOUT_S:-600}"
SCENARIO_ID="${SCENARIO_ID:-inference_scenario_d_general}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/senpai/research/baselines}"

mkdir -p "$LOG_DIR"

export HF_HOME="${HF_HOME:-${HOME}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$REPO_ROOT"

SERVER_LOG="$LOG_DIR/pytorch_baseline_D_server.log"
PRECOMPUTE_LOG="$LOG_DIR/pytorch_baseline_D_precompute.log"

echo "=== Starting transformers PyTorch baseline server on port $PORT ==="
echo "Model: $MODEL_ID  Scenario: $SCENARIO_ID  Request limit: $REQUEST_LIMIT"
echo "Logs: $SERVER_LOG, $PRECOMPUTE_LOG"

"$VENV/bin/python" -m src.eval.inference.servers.transformers_openai_server \
    --model "$MODEL_ID" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --dtype bfloat16 \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "SERVER_PID=$SERVER_PID"
echo "$SERVER_PID" > "$LOG_DIR/.server.pid"

echo "Waiting for /v1/models on port $PORT (up to 180s)..."
for i in $(seq 1 36); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "Server ready after ${i}x5s."
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server PID $SERVER_PID died early. Tail:"
        tail -n 30 "$SERVER_LOG"
        exit 1
    fi
    sleep 5
done

if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "Server failed to come up in 180s. Tail:"
    tail -n 50 "$SERVER_LOG"
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

echo "=== Running precompute_baseline.py (limit=$REQUEST_LIMIT, concurrency=1) ==="
"$VENV/bin/python" -m src.eval.inference.precompute_baseline \
    --scenario-id "$SCENARIO_ID" \
    --base-model "$MODEL_ID" \
    --server-url "http://127.0.0.1:$PORT" \
    --concurrency-override 1 \
    --request-limit "$REQUEST_LIMIT" \
    --request-timeout-s "$REQUEST_TIMEOUT_S" \
    2>&1 | tee "$PRECOMPUTE_LOG"

EXIT_CODE=${PIPESTATUS[0]}
echo "precompute exit code: $EXIT_CODE"

echo "=== Stopping baseline server ==="
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "Precompute failed; not copying baseline file."
    exit "$EXIT_CODE"
fi

# Unwrap precompute_baseline.py's wrapped output into a summarize_metrics-compatible JSON.
# precompute_baseline.py stores {"generated_at_unix": ..., "baseline": {profiles: {burst: ...}}};
# summarize_metrics.py / log_metrics_to_wandb.py expect `profiles` at the top level (same shape
# as evaluate.py's metrics.json). We also stash the original wrapper alongside for provenance.
MODEL_SAFE=$(echo "$MODEL_ID" | sed 's/[^A-Za-z0-9_.-]/_/g' | sed 's/^_//;s/_$//')
SRC_METRICS="$REPO_ROOT/src/eval/inference/baselines/speed/default/$SCENARIO_ID/$MODEL_SAFE/baseline_metrics.json"
DST_METRICS="$LOG_DIR/pytorch_baseline_D_blackwell.json"
DST_RAW="$LOG_DIR/pytorch_baseline_D_blackwell.precompute.json"

if [ -f "$SRC_METRICS" ]; then
    cp "$SRC_METRICS" "$DST_RAW"
    "$VENV/bin/python" - <<PY
import json, sys
src = json.loads(open("$SRC_METRICS", encoding="utf-8").read())
baseline = src.get("baseline") if isinstance(src, dict) else None
if not isinstance(baseline, dict) or "profiles" not in baseline:
    print("Unexpected baseline JSON shape; expected {'baseline': {'profiles': ...}}.", file=sys.stderr)
    sys.exit(2)
# Attach provenance to the summarize-compatible blob.
baseline.setdefault("_provenance", {})
baseline["_provenance"].update({
    "source": "precompute_baseline.py",
    "scenario_id": src.get("scenario_id"),
    "base_model": src.get("base_model"),
    "server_url": src.get("server_url"),
    "requests_sha256": src.get("requests_sha256"),
    "generated_at_unix": src.get("generated_at_unix"),
    "request_limit_applied": $REQUEST_LIMIT,
    "concurrency_override": 1,
    "hardware_note": "Blackwell sm_120 (RTX PRO 6000), bfloat16 transformers backend",
})
open("$DST_METRICS", "w", encoding="utf-8").write(json.dumps(baseline, indent=2))
print(f"Wrote summarize-compatible baseline to $DST_METRICS")
PY
    ls -la "$DST_METRICS" "$DST_RAW"
else
    echo "Expected baseline_metrics.json not found at: $SRC_METRICS"
    find "$REPO_ROOT/src/eval/inference/baselines/speed/default/$SCENARIO_ID" -name "baseline_metrics.json" 2>/dev/null | head -5
    exit 1
fi
