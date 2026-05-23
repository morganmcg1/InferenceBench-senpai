#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

JOB_NAME="inferencebench-scoring-assets-$(date -u +%Y%m%d%H%M%S)"
REPO_URL="$(git config --get remote.origin.url)"
REPO_BRANCH="$(git branch --show-current)"
IMAGE="${INFERENCE_BENCH_SENPAI_IMAGE:-ghcr.io/morganmcg1/inferencebench-senpai:pr-1}"
IMAGE_PULL_SECRET="${INFERENCE_BENCH_IMAGE_PULL_SECRET:-ghcr-morganmcg1-pull}"
PVC_CLAIM_NAME="${INFERENCE_BENCH_PVC_CLAIM_NAME:-new-pvc}"
PVC_MOUNT_PATH="${INFERENCE_BENCH_PVC_MOUNT_PATH:-/mnt/new-pvc}"
EXPORT_SLUG="${INFERENCE_BENCH_SCORING_ASSET_SLUG:-rtxpro6000-seed248}"
SCENARIO_ARG="${INFERENCE_BENCH_SENPAI_SCENARIOS:-all}"
EXPECTED_GPU="${INFERENCE_BENCH_EXPECTED_GPU:-RTX PRO 6000}"
WAIT=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: senpai/run_scoring_setup_job.sh [options]

Launch a one-GPU Kubernetes setup job that clones this target branch, runs
senpai/prepare_scoring_assets.sh, and exports the generated baselines to the
shared PVC. Run this before starting the two-hour SENPAI optimization clock.

Options:
  --job-name NAME              Kubernetes Job name
  --repo-url URL               Target repo URL (default: current origin)
  --repo-branch BRANCH         Target repo branch (default: current branch)
  --image IMAGE                Container image with SENPAI + InferenceBench deps
  --image-pull-secret NAME     Pull secret (default: ghcr-morganmcg1-pull)
  --pvc NAME                   Shared PVC claim (default: new-pvc)
  --pvc-mount-path PATH        PVC mount path in the job (default: /mnt/new-pvc)
  --export-slug SLUG           Assets written under $PVC/inferencebench-senpai/scoring-assets/SLUG
  --scenario A,B,C,D|all       Scenarios to prepare (default: all)
  --expected-gpu TEXT          GPU substring required by preflight (default: RTX PRO 6000)
  --no-wait                    Apply the job and return immediately
  --dry-run                    Render/validate the manifest locally; do not apply
  -h, --help                   Show this help
EOF
}

while (($#)); do
  case "$1" in
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --repo-url) REPO_URL="$2"; shift 2 ;;
    --repo-branch) REPO_BRANCH="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --image-pull-secret) IMAGE_PULL_SECRET="$2"; shift 2 ;;
    --pvc) PVC_CLAIM_NAME="$2"; shift 2 ;;
    --pvc-mount-path) PVC_MOUNT_PATH="$2"; shift 2 ;;
    --export-slug) EXPORT_SLUG="$2"; shift 2 ;;
    --scenario) SCENARIO_ARG="$2"; shift 2 ;;
    --expected-gpu) EXPECTED_GPU="$2"; shift 2 ;;
    --no-wait) WAIT=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$REPO_URL" || -z "$REPO_BRANCH" ]]; then
  echo "repo URL and branch are required" >&2
  exit 2
fi

EXPORT_DIR="$PVC_MOUNT_PATH/inferencebench-senpai/scoring-assets/$EXPORT_SLUG"

manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT

cat >"$manifest" <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: $JOB_NAME
  labels:
    app: inferencebench-scoring-assets
    scoring-asset-slug: $EXPORT_SLUG
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app: inferencebench-scoring-assets
        scoring-asset-slug: $EXPORT_SLUG
    spec:
      restartPolicy: Never
      tolerations:
        - key: is_gpu
          operator: Exists
          effect: PreferNoSchedule
      imagePullSecrets:
        - name: $IMAGE_PULL_SECRET
      containers:
        - name: prepare
          image: $IMAGE
          imagePullPolicy: Always
          resources:
            requests:
              cpu: "16"
              memory: 120Gi
              nvidia.com/gpu: "1"
            limits:
              cpu: "16"
              memory: 120Gi
              nvidia.com/gpu: "1"
          env:
            - name: REPO_URL
              value: "$REPO_URL"
            - name: REPO_BRANCH
              value: "$REPO_BRANCH"
            - name: EXPORT_DIR
              value: "$EXPORT_DIR"
            - name: SCENARIO_ARG
              value: "$SCENARIO_ARG"
            - name: EXPECTED_GPU
              value: "$EXPECTED_GPU"
            - name: GITHUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: senpai-secrets
                  key: github-token
                  optional: true
            - name: WANDB_API_KEY
              valueFrom:
                secretKeyRef:
                  name: senpai-secrets
                  key: wandb-api-key
                  optional: true
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: kagent-secrets
                  key: hf-token
                  optional: true
          volumeMounts:
            - name: shared-pvc
              mountPath: $PVC_MOUNT_PATH
          command: ["/bin/bash", "-lc"]
          args:
            - |
              set -euo pipefail
              mkdir -p /workspace
              rm -rf /workspace/inferencebench
              if [[ -n "\${GITHUB_TOKEN:-}" ]]; then
                git -c http.extraHeader="Authorization: Bearer \${GITHUB_TOKEN}" \
                  clone --branch "\$REPO_BRANCH" --single-branch "\$REPO_URL" /workspace/inferencebench
              else
                git clone --branch "\$REPO_BRANCH" --single-branch "\$REPO_URL" /workspace/inferencebench
              fi
              cd /workspace/inferencebench
              git config --global --add safe.directory /workspace/inferencebench
              echo "[setup-job] head=\$(git rev-parse --short HEAD) branch=\$(git branch --show-current)"
              senpai/prepare_scoring_assets.sh \
                --scenario "\$SCENARIO_ARG" \
                --expected-gpu "\$EXPECTED_GPU" \
                --export-dir "\$EXPORT_DIR"
      volumes:
        - name: shared-pvc
          persistentVolumeClaim:
            claimName: $PVC_CLAIM_NAME
EOF

echo "[setup-job] applying $JOB_NAME"
if [[ "$DRY_RUN" == "1" ]]; then
  kubectl apply --dry-run=client -f "$manifest"
  exit 0
fi

kubectl apply -f "$manifest"
echo "[setup-job] export dir: $EXPORT_DIR"

if [[ "$WAIT" == "1" ]]; then
  kubectl wait --for=condition=complete "job/$JOB_NAME" --timeout=24h
  kubectl logs "job/$JOB_NAME"
fi
