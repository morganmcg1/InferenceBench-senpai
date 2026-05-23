#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_MODEL="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
SCENARIO_ARG="all"
EXPECTED_GPU="${INFERENCE_BENCH_EXPECTED_GPU:-RTX PRO 6000}"
BACKEND="${INFERENCE_BENCH_SPEED_BASELINE_BACKEND:-torch}"
QUALITY_BACKEND="${INFERENCE_BENCH_QUALITY_BASELINE_BACKEND:-torch}"
DATASET_SEED="${INFERENCE_BENCH_DATASET_SEED:-248}"
QUALITY_SEED="${INFERENCE_BENCH_QUALITY_SEED:-$DATASET_SEED}"
MMLUPRO_N="${INFERENCE_BENCH_QUALITY_MMLUPRO_N:-500}"
MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"
SERVER_START_TIMEOUT_S="${INFERENCE_BENCH_SERVER_START_TIMEOUT_S:-900}"
RUNTIME_CACHE_DIR="${INFERENCE_BENCH_RUNTIME_CACHE_DIR:-${TMPDIR:-/tmp}/inferencebench-scoring-cache}"
EXPORT_DIR=""
LEADERBOARD_MODE=0
REQUIRE_WANDB=1
RUN_PRECOMPUTE=1
EXPORTED_ON_EXIT=0

export_assets() {
  local exit_code="$1"
  if [[ -z "$EXPORT_DIR" || ! -d src/eval/inference/baselines ]]; then
    return 0
  fi
  mkdir -p "$EXPORT_DIR"
  rm -rf "$EXPORT_DIR/baselines"
  cp -a src/eval/inference/baselines "$EXPORT_DIR/baselines"
  {
    echo "generated_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repo_head=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "base_model=$BASE_MODEL"
    echo "scenarios=$(IFS=,; echo "${scenario_letters[*]:-}")"
    echo "backend=$BACKEND"
    echo "quality_backend=$QUALITY_BACKEND"
    echo "dataset_seed=$DATASET_SEED"
    echo "quality_seed=$QUALITY_SEED"
    echo "mmlupro_n=$MMLUPRO_N"
    echo "expected_gpu=$EXPECTED_GPU"
    echo "exit_code=$exit_code"
  } > "$EXPORT_DIR/manifest.env"
  EXPORTED_ON_EXIT=1
  echo "[prepare] exported scoring assets to $EXPORT_DIR (exit_code=$exit_code)"
}

on_exit() {
  local exit_code="$?"
  if [[ "$EXPORTED_ON_EXIT" == "0" ]]; then
    export_assets "$exit_code" || true
  fi
  exit "$exit_code"
}

trap on_exit EXIT

usage() {
  cat <<'EOF'
Usage: senpai/prepare_scoring_assets.sh [options]

Generate the scoring assets SENPAI needs before a two-hour InferenceBench run:
deterministic speed requests, PyTorch speed baselines, MMLU-Pro quality samples,
and the PyTorch quality baseline registry.

Options:
  --scenario A,B,C,D|all       Scenarios to prepare (default: all)
  --expected-gpu TEXT          GPU substring required by preflight (default: RTX PRO 6000)
  --leaderboard-mode           Require H100-class preflight checks
  --base-model MODEL           Base model id (default: mistralai/Mistral-7B-Instruct-v0.3)
  --backend NAME               Speed baseline backend namespace (default: torch)
  --quality-backend NAME       Quality baseline backend namespace (default: torch)
  --dataset-seed N             Speed/request seed (default: 248)
  --quality-seed N             Quality seed (default: dataset seed)
  --mmlupro-n N                MMLU-Pro quality sample count (default: 500)
  --max-model-len N            Server max model length (default: 32768)
  --runtime-cache-dir DIR      Runtime cache dir for HF/Triton/Inductor
  --server-start-timeout-s N   Baseline server readiness timeout (default: 900)
  --export-dir DIR             Also copy src/eval/inference/baselines to DIR/baselines
  --preflight-only             Do not regenerate assets; only run the hard preflight
  --no-wandb                   Do not require WANDB_API_KEY in the final preflight
  -h, --help                   Show this help

This setup is intentionally outside the SENPAI optimization clock.
EOF
}

while (($#)); do
  case "$1" in
    --scenario) SCENARIO_ARG="$2"; shift 2 ;;
    --expected-gpu) EXPECTED_GPU="$2"; shift 2 ;;
    --leaderboard-mode) LEADERBOARD_MODE=1; EXPECTED_GPU="H100"; shift ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --quality-backend) QUALITY_BACKEND="$2"; shift 2 ;;
    --dataset-seed) DATASET_SEED="$2"; shift 2 ;;
    --quality-seed) QUALITY_SEED="$2"; shift 2 ;;
    --mmlupro-n) MMLUPRO_N="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --runtime-cache-dir) RUNTIME_CACHE_DIR="$2"; shift 2 ;;
    --server-start-timeout-s) SERVER_START_TIMEOUT_S="$2"; shift 2 ;;
    --export-dir) EXPORT_DIR="$2"; shift 2 ;;
    --preflight-only) RUN_PRECOMPUTE=0; shift ;;
    --no-wandb) REQUIRE_WANDB=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

scenario_folders=()
scenario_letters=()
IFS=',' read -r -a raw_scenarios <<<"$SCENARIO_ARG"
if [[ "${SCENARIO_ARG,,}" == "all" ]]; then
  scenario_letters=(A B C D)
  scenario_folders=(
    inference_scenario_a_input_heavy
    inference_scenario_b_output_heavy
    inference_scenario_c_high_load
    inference_scenario_d_general
  )
else
  for raw in "${raw_scenarios[@]}"; do
    key="$(echo "$raw" | tr '[:lower:]' '[:upper:]' | xargs)"
    case "$key" in
      A) scenario_letters+=(A); scenario_folders+=(inference_scenario_a_input_heavy) ;;
      B) scenario_letters+=(B); scenario_folders+=(inference_scenario_b_output_heavy) ;;
      C) scenario_letters+=(C); scenario_folders+=(inference_scenario_c_high_load) ;;
      D) scenario_letters+=(D); scenario_folders+=(inference_scenario_d_general) ;;
      *) echo "unknown scenario '$raw'; use A,B,C,D,all" >&2; exit 2 ;;
    esac
  done
fi

mkdir -p "$RUNTIME_CACHE_DIR"
export INFERENCE_BENCH_BASE_MODEL="$BASE_MODEL"
export INFERENCE_BENCH_DATASET_SEED="$DATASET_SEED"
export INFERENCE_BENCH_QUALITY_SEED="$QUALITY_SEED"
export INFERENCE_BENCH_QUALITY_MMLUPRO_N="$MMLUPRO_N"
export INFERENCE_BENCH_MAX_MODEL_LEN="$MAX_MODEL_LEN"
export INFERENCE_BENCH_RUNTIME_CACHE_DIR="$RUNTIME_CACHE_DIR"
export INFERENCE_BENCH_ALLOW_HF_DOWNLOAD="${INFERENCE_BENCH_ALLOW_HF_DOWNLOAD:-1}"
export INFERENCE_BENCH_QUALITY_BASELINE_BACKEND="$QUALITY_BACKEND"

if [[ -f senpai/runtime_env.sh ]]; then
  # shellcheck disable=SC1091
  source senpai/runtime_env.sh
fi

echo "[prepare] repo=$ROOT_DIR"
echo "[prepare] model=$BASE_MODEL scenarios=${scenario_letters[*]} backend=$BACKEND quality_backend=$QUALITY_BACKEND"
echo "[prepare] cache=$RUNTIME_CACHE_DIR"

if [[ "$RUN_PRECOMPUTE" == "1" ]]; then
  python -m src.eval.inference.precompute_all_baselines \
    --cache-quality-samples \
    --backends "$BACKEND" \
    --base-model "$BASE_MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dataset-seed "$DATASET_SEED" \
    --quality-seed "$QUALITY_SEED" \
    --mmlupro-n "$MMLUPRO_N" \
    --quality-concurrency 1 \
    --runtime-cache-dir "$RUNTIME_CACHE_DIR" \
    --server-start-timeout-s "$SERVER_START_TIMEOUT_S" \
    --scenarios "${scenario_folders[@]}"
fi

preflight_args=(
  senpai/preflight.py
  --scenario "$(IFS=,; echo "${scenario_letters[*]}")"
  --expected-gpu "$EXPECTED_GPU"
  --base-model "$BASE_MODEL"
  --dataset-seed "$DATASET_SEED"
  --quality-seed "$QUALITY_SEED"
  --mmlupro-n "$MMLUPRO_N"
  --speed-baseline-backend "$BACKEND"
  --quality-baseline-backend "$QUALITY_BACKEND"
)
if [[ "$LEADERBOARD_MODE" == "1" ]]; then
  preflight_args+=(--leaderboard-mode)
fi
if [[ "$REQUIRE_WANDB" == "1" ]]; then
  preflight_args+=(--require-wandb)
fi

python "${preflight_args[@]}"

if [[ -n "$EXPORT_DIR" ]]; then
  mkdir -p "$EXPORT_DIR"
  python "${preflight_args[@]}" --json-output "$EXPORT_DIR/preflight.json"
  if [[ "$EXPORTED_ON_EXIT" == "0" ]]; then
    export_assets 0
  fi
fi
