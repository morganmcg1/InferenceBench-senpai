#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SCENARIO_ARG="${INFERENCE_BENCH_SENPAI_SCENARIOS:-all}"
EXPECTED_GPU="${INFERENCE_BENCH_EXPECTED_GPU:-RTX PRO 6000}"
BASE_MODEL="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
DATASET_SEED="${INFERENCE_BENCH_DATASET_SEED:-248}"
QUALITY_SEED="${INFERENCE_BENCH_QUALITY_SEED:-$DATASET_SEED}"
MMLUPRO_N="${INFERENCE_BENCH_QUALITY_MMLUPRO_N:-500}"
SPEED_BACKEND="${INFERENCE_BENCH_SPEED_BASELINE_BACKEND:-torch}"
QUALITY_BACKEND="${INFERENCE_BENCH_QUALITY_BASELINE_BACKEND:-torch}"
LEADERBOARD_MODE=0
REQUIRE_WANDB=1
IMPORT_DIR="${INFERENCE_BENCH_SCORING_ASSETS_DIR:-}"

usage() {
  cat <<'EOF'
Usage: senpai/require_scoring_preflight.sh [options]

Fail fast unless this clone has the deterministic request files, PyTorch speed
baselines, MMLU-Pro quality samples, quality registry, W&B, and expected GPU
needed for SENPAI to report InferenceBench speedup metrics.

Options:
  --scenario A,B,C,D|all       Scenarios to check (default: all)
  --expected-gpu TEXT          GPU substring required by preflight (default: RTX PRO 6000)
  --leaderboard-mode           Require H100-class preflight checks
  --base-model MODEL           Base model id
  --import-dir DIR             Copy DIR/baselines into src/eval/inference/baselines first
  --no-wandb                   Do not require WANDB_API_KEY
  -h, --help                   Show this help
EOF
}

while (($#)); do
  case "$1" in
    --scenario) SCENARIO_ARG="$2"; shift 2 ;;
    --expected-gpu) EXPECTED_GPU="$2"; shift 2 ;;
    --leaderboard-mode) LEADERBOARD_MODE=1; EXPECTED_GPU="H100"; shift ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --import-dir) IMPORT_DIR="$2"; shift 2 ;;
    --no-wandb) REQUIRE_WANDB=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -n "$IMPORT_DIR" ]]; then
  if [[ ! -d "$IMPORT_DIR/baselines" ]]; then
    echo "scoring asset import dir is missing baselines/: $IMPORT_DIR" >&2
    exit 1
  fi
  rm -rf src/eval/inference/baselines
  mkdir -p src/eval/inference
  cp -a "$IMPORT_DIR/baselines" src/eval/inference/baselines
  echo "[preflight] imported scoring assets from $IMPORT_DIR"
fi

args=(
  senpai/preflight.py
  --scenario "$SCENARIO_ARG"
  --expected-gpu "$EXPECTED_GPU"
  --base-model "$BASE_MODEL"
  --dataset-seed "$DATASET_SEED"
  --quality-seed "$QUALITY_SEED"
  --mmlupro-n "$MMLUPRO_N"
  --speed-baseline-backend "$SPEED_BACKEND"
  --quality-baseline-backend "$QUALITY_BACKEND"
)
if [[ "$LEADERBOARD_MODE" == "1" ]]; then
  args+=(--leaderboard-mode)
fi
if [[ "$REQUIRE_WANDB" == "1" ]]; then
  args+=(--require-wandb)
fi

python "${args[@]}"
