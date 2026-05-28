#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

JOB_NAME="inferencebench-launch-smoke-$(date -u +%Y%m%d%H%M%S)"
REPO_BRANCH="$(git branch --show-current)"
UPSTREAM_REMOTE="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null | cut -d/ -f1 || true)"
if [[ -n "$UPSTREAM_REMOTE" ]]; then
  REPO_URL="$(git config --get "remote.${UPSTREAM_REMOTE}.url")"
else
  REPO_URL="$(git config --get remote.origin.url)"
fi
IMAGE="${INFERENCE_BENCH_SENPAI_IMAGE:-ghcr.io/morganmcg1/inferencebench-senpai:pr-1}"
IMAGE_PULL_SECRET="${INFERENCE_BENCH_IMAGE_PULL_SECRET:-ghcr-morganmcg1-pull}"
PVC_CLAIM_NAME="${INFERENCE_BENCH_PVC_CLAIM_NAME:-new-pvc}"
PVC_MOUNT_PATH="${INFERENCE_BENCH_PVC_MOUNT_PATH:-/mnt/new-pvc}"
IMPORT_DIR="${INFERENCE_BENCH_SCORING_ASSETS_DIR:-$PVC_MOUNT_PATH/inferencebench-senpai/scoring-assets/rtxpro6000-seed248}"
SCENARIO_ARG="${INFERENCE_BENCH_SENPAI_SCENARIOS:-all}"
EXPECTED_GPU="${INFERENCE_BENCH_EXPECTED_GPU:-RTX PRO 6000}"
SECRET_NAME="${INFERENCE_BENCH_SENPAI_SECRET_NAME:-senpai-secrets}"
WAIT=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: senpai/run_launch_smoke_test_job.sh [options]

Launch a one-GPU Kubernetes smoke job that uses the exact SENPAI image and
target branch intended for a run. Use it before arming the 2 hour cutoff gate.

The job checks:
  - GitHub/W&B/Anthropic secrets are mounted under the expected env names.
  - Claude Code is installed and its JSON config can be initialized/parsed.
  - Weave Claude plugin is installed and its settings JSON is valid.
  - `command -v nvidia-smi` resolves to a non-empty executable and works.
  - `senpai/require_scoring_preflight.sh` passes with the prepared scoring assets.

Options:
  --job-name NAME              Kubernetes Job name
  --repo-url URL               Target repo URL (default: current origin)
  --repo-branch BRANCH         Target repo branch (default: current branch)
  --image IMAGE                Container image to test
  --image-pull-secret NAME     Pull secret (default: ghcr-morganmcg1-pull)
  --pvc NAME                   Shared PVC claim (default: new-pvc)
  --pvc-mount-path PATH        PVC mount path in the job (default: /mnt/new-pvc)
  --import-dir DIR             Prepared scoring assets dir on the PVC
  --scenario A,B,C,D|all       Scenarios to preflight (default: all)
  --expected-gpu TEXT          GPU substring required by preflight (default: RTX PRO 6000)
  --secret-name NAME           Secret with github-token/wandb-api-key/anthropic-api-key
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
    --import-dir) IMPORT_DIR="$2"; shift 2 ;;
    --scenario) SCENARIO_ARG="$2"; shift 2 ;;
    --expected-gpu) EXPECTED_GPU="$2"; shift 2 ;;
    --secret-name) SECRET_NAME="$2"; shift 2 ;;
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

if [[ "$REPO_URL" == git@github.com:* ]]; then
  REPO_URL="https://github.com/${REPO_URL#git@github.com:}"
fi

manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT

cat >"$manifest" <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: $JOB_NAME
  labels:
    app: inferencebench-launch-smoke
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app: inferencebench-launch-smoke
    spec:
      restartPolicy: Never
      tolerations:
        - key: is_gpu
          operator: Exists
          effect: PreferNoSchedule
      imagePullSecrets:
        - name: $IMAGE_PULL_SECRET
      containers:
        - name: smoke
          image: $IMAGE
          imagePullPolicy: Always
          resources:
            requests:
              cpu: "8"
              memory: 80Gi
              nvidia.com/gpu: "1"
            limits:
              cpu: "8"
              memory: 80Gi
              nvidia.com/gpu: "1"
          env:
            - name: REPO_URL
              value: "$REPO_URL"
            - name: REPO_BRANCH
              value: "$REPO_BRANCH"
            - name: IMPORT_DIR
              value: "$IMPORT_DIR"
            - name: SCENARIO_ARG
              value: "$SCENARIO_ARG"
            - name: EXPECTED_GPU
              value: "$EXPECTED_GPU"
            - name: GITHUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: $SECRET_NAME
                  key: github-token
            - name: WANDB_API_KEY
              valueFrom:
                secretKeyRef:
                  name: $SECRET_NAME
                  key: wandb-api-key
            - name: ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: $SECRET_NAME
                  key: anthropic-api-key
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
              : "\${GITHUB_TOKEN:?missing github-token in $SECRET_NAME}"
              : "\${WANDB_API_KEY:?missing wandb-api-key in $SECRET_NAME}"
              : "\${ANTHROPIC_API_KEY:?missing anthropic-api-key in $SECRET_NAME}"

              echo "[smoke] image=$IMAGE"
              echo "[smoke] repo=\$REPO_URL branch=\$REPO_BRANCH"
              echo "[smoke] import_dir=\$IMPORT_DIR scenario=\$SCENARIO_ARG expected_gpu=\$EXPECTED_GPU"

              smi_path="\$(command -v nvidia-smi || true)"
              if [[ -z "\$smi_path" ]]; then
                echo "[smoke] ERROR: nvidia-smi is not on PATH" >&2
                exit 21
              fi
              if [[ ! -s "\$smi_path" ]]; then
                ls -l "\$smi_path" >&2 || true
                echo "[smoke] ERROR: nvidia-smi path is empty or non-readable: \$smi_path" >&2
                exit 22
              fi
              echo "[smoke] nvidia-smi path=\$smi_path"
              nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

              command -v claude
              claude --version
              mkdir -p "\$HOME/.claude"
              if [[ -f "\$HOME/.claude.json" ]]; then
                python3 -m json.tool "\$HOME/.claude.json" >/dev/null
              else
                printf '{}\n' > "\$HOME/.claude.json"
              fi
              python3 -m json.tool "\$HOME/.claude.json" >/dev/null
              echo "[smoke] claude config OK"

              command -v weave-claude-plugin
              test -s "\$HOME/.weave_claude_plugin/settings.json"
              python3 -m json.tool "\$HOME/.weave_claude_plugin/settings.json" >/dev/null
              echo "[smoke] weave plugin config OK"

              mkdir -p /workspace
              rm -rf /workspace/inferencebench
              umask 077
              printf 'https://x-access-token:%s@github.com\n' "\$GITHUB_TOKEN" > /tmp/git-credentials
              git -c credential.helper='store --file=/tmp/git-credentials' \
                clone --branch "\$REPO_BRANCH" --single-branch "\$REPO_URL" /workspace/inferencebench
              rm -f /tmp/git-credentials
              cd /workspace/inferencebench
              git config --global --add safe.directory /workspace/inferencebench
              echo "[smoke] head=\$(git rev-parse --short HEAD) branch=\$(git branch --show-current)"

              source senpai/runtime_env.sh
              python senpai/runtime_doctor.py
              senpai/require_scoring_preflight.sh \
                --import-dir "\$IMPORT_DIR" \
                --scenario "\$SCENARIO_ARG" \
                --expected-gpu "\$EXPECTED_GPU"
              echo "[smoke] PASS"
      volumes:
        - name: shared-pvc
          persistentVolumeClaim:
            claimName: $PVC_CLAIM_NAME
EOF

echo "[smoke-job] applying $JOB_NAME"
if [[ "$DRY_RUN" == "1" ]]; then
  kubectl apply --dry-run=client -f "$manifest"
  exit 0
fi

kubectl apply -f "$manifest"

if [[ "$WAIT" == "1" ]]; then
  kubectl wait --for=condition=complete "job/$JOB_NAME" --timeout=30m
  kubectl logs "job/$JOB_NAME"
  image_id="$(kubectl get pods -l job-name="$JOB_NAME" -o jsonpath='{.items[0].status.containerStatuses[0].imageID}' 2>/dev/null || true)"
  if [[ "$image_id" == *@sha256:* ]]; then
    echo "[smoke-job] image digest passed: ${image_id#docker-pullable://}"
    echo "[smoke-job] use this immutable digest for timed launches instead of a mutable tag."
  fi
fi
