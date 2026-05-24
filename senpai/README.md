# SENPAI Helpers For InferenceBench

This directory is intentionally separate from the official benchmark harness.
Use it for SENPAI-only launcher recipes, result summarizers, research notes, and
future integration helpers.

Suggested layout:

```text
senpai/
  launchers/<scenario>/<slug>/start_server.sh
  research/
  summarize_metrics.py
  log_metrics_to_wandb.py
```

Normal experiment PRs should prefer adding or modifying files here rather than
touching `src/eval`, `src/run_task.sh`, `containers`, or scenario definitions.
The official final benchmark still measures the task-local `./start_server.sh`
after a clean supervised relaunch.

W&B logging for SENPAI runs should go to:

```text
wandb-applied-ai-team/inferencebench-senpai
```

Use `log_metrics_to_wandb.py` on official `metrics.json` files. It is a logging
wrapper only; it does not change benchmark evaluation.

## Preflight

Before a launch, run the mode that matches the hardware:

```bash
# Current RTX PRO 6000 shakedown mode
senpai/require_scoring_preflight.sh --scenario all \
  --expected-gpu "RTX PRO 6000"

# Later H100 leaderboard-comparable mode
senpai/require_scoring_preflight.sh --leaderboard-mode --scenario all \
  --expected-gpu H100
```

The check fails if deterministic requests, PyTorch speed baselines, MMLU-Pro
quality samples, the PyTorch quality registry, W&B, or the requested hardware
are missing. It also imports torch and vLLM inside the target image so ABI or
dependency drift is caught before the optimization clock starts. Fix those
before assigning serving PRs. RTX PRO 6000 results are for orchestration and
search-direction shakedown; repeat winners on H100 before claiming README
leaderboard wins.

To build the required scoring assets on the current RTX PRO 6000 cluster,
outside the two-hour SENPAI optimization clock:

```bash
senpai/run_scoring_setup_job.sh \
  --repo-branch codex/inferencebench-senpai-target \
  --export-slug rtxpro6000-seed248 \
  --scenario all \
  --expected-gpu "RTX PRO 6000"
```

The job exports assets under
`/mnt/new-pvc/inferencebench-senpai/scoring-assets/<slug>/`. Import and verify
those assets in any launch clone with:

```bash
senpai/require_scoring_preflight.sh \
  --import-dir /mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248 \
  --scenario all \
  --expected-gpu "RTX PRO 6000"
```

Do not open the SENPAI start gate until this hard preflight passes in the same
target branch, image, and hardware context the run will use.

If tokenizer/runtime drift prevents the official evaluator from sampling
LongBench-v2 requests, materialize request files once without editing
`src/eval`:

```bash
INFERENCE_BENCH_ALLOW_HF_DOWNLOAD=1 \
python senpai/materialize_requests.py --scenario all --backend torch
```

For pod-local experiments, stage a task workspace that mirrors the official
harness:

```bash
python senpai/create_task_workspace.py --scenario A \
  --output /tmp/inferencebench-A --starting-point vllm_running
```

The workspace includes `task/eval_env.sh`,
`task/clean_eval_artifacts.sh`, `task/test_server.sh`, and `task/evaluate.py`.
Source `eval_env.sh` before manual commands; it pins the base model, scenario,
request file, PyTorch baseline metrics path, quality registry, and runtime
cache setup for the copied evaluator.

## Runtime Environment

In shared RTX PRO 6000 shakedown pods, source the runtime helper before
launching serving backends:

```bash
source senpai/runtime_env.sh
```

It exports CUDA pip-package include/library paths and pod-local JIT cache
locations for vLLM, FlashInfer, Triton, and related backends. On RTX PRO 6000
shakedown pods it also defaults `INFERENCE_BENCH_MAX_MODEL_LEN=32768` and
disables vLLM's implicit FlashInfer sampler/prefill path unless a launcher opts
back in. This is a process-local launch aid, not a benchmark harness change.

## Shared GPU Slot

In one-GPU, multi-student pods, use `gpu_slot.py` to serialize heavy benchmark
work without adding a separate runner:

```bash
python senpai/gpu_slot.py status --json
python senpai/gpu_slot.py run --wait --ttl-s 1800 \
  --owner "$STUDENT_NAME" --pr 123 --scenario C -- \
  bash -lc 'cd /tmp/inferencebench-C/task && source ./eval_env.sh && ./clean_eval_artifacts.sh && ./test_server.sh > agent/server.log 2>&1 & server_pid=$!; trap "kill $server_pid 2>/dev/null || true" EXIT; python evaluate.py --json-output-file metrics_full.json'
```

For queued work, prefer `gpu_slot.py run --wait ...`; do not parse exact
`status` text in shell loops. The status line is intentionally human-facing.

The lease records a unique lease ID, owner, PR, scenario, command, timestamps,
and expiry in `/tmp/inferencebench-gpu-slot.json` by default. `run` heartbeats
the lease, refuses to acquire a free-looking slot when `nvidia-smi` still shows
unleased compute processes, and terminates the command process group if the
lease is lost. This prevents accidental overlapping full workloads while
preserving the official `test_server.sh` plus `evaluate.py` evaluation path.

## Cluster Cutoff And Conversation Logs

Use `arm_cluster_cutoff.sh` when starting Kubernetes SENPAI runs. It waits for
the expected pods, starts the wall-clock budget, harvests `.claude` from root
and per-student homes plus SENPAI student logs from each tagged pod to the PVC
before shutdown, deletes the tagged SENPAI deployments/configmaps/secrets, and
starts a best-effort local mirror into `conversation_logs/`.

```bash
senpai/arm_cluster_cutoff.sh \
  --run-slug ib-YYYYMMDD-rerun \
  --tags-csv ib-YYYYMMDD-r1,ib-YYYYMMDD-r2,ib-YYYYMMDD-r3,ib-YYYYMMDD-r4,ib-YYYYMMDD-r5 \
  --expected-pods 10 \
  --expected-deployments 10 \
  --budget-hours 2 \
  --harvest-lead-seconds 300 \
  --start-gate-path /mnt/new-pvc/senpai-start-gates/ib-YYYYMMDD-rerun/start \
  --image ghcr.io/morganmcg1/inferencebench-senpai:pr-1 \
  --image-pull-secret ghcr-morganmcg1-pull
```

Pass the same path to SENPAI's `k8s/launch.py` as `--start_gate_path` so the
advisor and student pods wait for the cutoff job to open the run.
