#!/usr/bin/env python3
"""Log InferenceBench metrics.json to W&B without modifying the benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import summarize_metrics


DEFAULT_ENTITY = "wandb-applied-ai-team"
DEFAULT_PROJECT = "inferencebench-senpai"


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            out.update(_flatten(child, child_prefix))
    elif isinstance(value, (int, float, str, bool)) or value is None:
        out[prefix] = value
    return out


def _wandb_config(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    config = {
        "scenario": result.get("scenario"),
        "base_model": args.base_model,
        "engine": args.engine,
        "launcher": args.launcher,
        "metrics_json": args.metrics_json,
        "baseline_metrics_json": args.baseline_metrics_json,
        "time_budget_hours": args.time_budget_hours,
    }
    return {key: value for key, value in config.items() if value not in (None, "")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", help="Path to InferenceBench metrics.json")
    parser.add_argument("--scenario", help="Scenario id: A, B, C, or D")
    parser.add_argument("--baseline-metrics-json", help="PyTorch baseline metrics.json for speedup reporting")
    parser.add_argument("--baseline-primary", type=float, help="Raw PyTorch baseline primary value for this scenario")
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name", help="W&B run name, e.g. $STUDENT_NAME/<slug>")
    parser.add_argument("--group", help="W&B group, usually the hypothesis or PR")
    parser.add_argument("--job-type", default="inferencebench-eval")
    parser.add_argument("--base-model")
    parser.add_argument("--engine")
    parser.add_argument("--launcher")
    parser.add_argument("--time-budget-hours", type=float)
    parser.add_argument("--offline", action="store_true", help="Use W&B offline mode")
    args = parser.parse_args()

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is not installed. Install it in the SENPAI image or target environment.") from exc

    summary_args = argparse.Namespace(
        metrics_json=args.metrics_json,
        scenario=args.scenario,
        baseline_metrics_json=args.baseline_metrics_json,
        baseline_primary=args.baseline_primary,
        wandb_run_id=[],
        status="complete",
        terminal=True,
        pending_arms=False,
    )
    result = summarize_metrics.build_result(summary_args)
    metrics = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))

    init_kwargs = {
        "entity": args.entity,
        "project": args.project,
        "name": args.name,
        "group": args.group,
        "job_type": args.job_type,
        "config": _wandb_config(args, result),
    }
    if args.offline:
        init_kwargs["mode"] = "offline"

    run = wandb.init(**init_kwargs)
    run_id = run.id
    result["wandb_run_ids"] = [run_id]
    log_data = _flatten(metrics)
    log_data.update(_flatten(result, "senpai"))
    primary = result["primary_metric"]
    raw_primary = result["raw_primary_metric"]
    log_data["senpai/primary_metric_value"] = primary["value"]
    log_data["senpai/raw_primary_metric_value"] = raw_primary["value"]
    if "speedup_over_pytorch" in result:
        log_data["senpai/speedup_over_pytorch"] = result["speedup_over_pytorch"]

    try:
        wandb.log(log_data)
        artifact = wandb.Artifact(
            name=f"inferencebench-metrics-{run_id}",
            type="inferencebench-metrics",
            metadata=result,
        )
        artifact.add_file(args.metrics_json)
        if args.baseline_metrics_json:
            artifact.add_file(args.baseline_metrics_json)
        run.log_artifact(artifact)
        run.summary["senpai_result"] = result
    finally:
        wandb.finish()

    print(f"SENPAI-WANDB-RUN: {run_id}")
    print("SENPAI-RESULT: " + json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
