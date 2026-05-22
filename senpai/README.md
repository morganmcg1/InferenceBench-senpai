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
