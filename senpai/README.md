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
python senpai/preflight.py --scenario all \
  --expected-gpu "RTX PRO 6000" \
  --require-wandb

# Later H100 leaderboard-comparable mode
python senpai/preflight.py --leaderboard-mode --scenario all \
  --expected-gpu H100 --require-wandb
```

The check fails if deterministic requests, PyTorch speed baselines, MMLU-Pro
quality samples, the PyTorch quality registry, W&B, or the requested hardware
are missing. Fix those before assigning serving PRs. RTX PRO 6000 results are
for orchestration and search-direction shakedown; repeat winners on H100 before
claiming README leaderboard wins.

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
