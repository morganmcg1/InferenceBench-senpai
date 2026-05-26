#!/usr/bin/env python3
"""Emit a SENPAI-RESULT marker from an InferenceBench metrics.json file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PRIMARY_NAMES = {
    "A": "scenario/A/inverse_ttft_p50",
    "B": "scenario/B/inverse_tpot_p50",
    "C": "scenario/C/geomean_request_throughput_req_per_s",
    "D": "scenario/D/geomean_inverse_latency_throughput",
}

FULL_QUALITY_N = 500


def _p50(profile: dict[str, Any], key: str) -> float:
    value = (profile.get(key) or {}).get("p50")
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"missing positive {key}.p50")
    return float(value)


def _throughput(profile: dict[str, Any]) -> float:
    value = profile.get("request_throughput_req_per_s")
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("missing positive request_throughput_req_per_s")
    return float(value)


def _geomean(values: list[float]) -> float:
    if not values or any(v <= 0 for v in values):
        raise ValueError("geomean requires positive values")
    return math.prod(values) ** (1.0 / len(values))


def infer_scenario(metrics: dict[str, Any], override: str | None) -> str:
    raw = (override or metrics.get("scenario") or "").strip().upper()
    if raw in {"A", "B", "C", "D"}:
        return raw
    aliases = {
        "INFERENCE_SCENARIO_A_INPUT_HEAVY": "A",
        "INFERENCE_SCENARIO_B_OUTPUT_HEAVY": "B",
        "INFERENCE_SCENARIO_C_HIGH_LOAD": "C",
        "INFERENCE_SCENARIO_D_GENERAL": "D",
    }
    key = raw.replace("-", "_")
    if key in aliases:
        return aliases[key]
    raise ValueError("scenario must be supplied as A, B, C, or D")


def primary_metric(metrics: dict[str, Any], scenario: str) -> tuple[str, float]:
    profiles = metrics.get("profiles") or {}
    burst = profiles.get("burst")
    if not isinstance(burst, dict):
        raise ValueError("missing burst profile")

    if scenario == "A":
        return PRIMARY_NAMES[scenario], 1.0 / _p50(burst, "ttft")
    if scenario == "B":
        return PRIMARY_NAMES[scenario], 1.0 / _p50(burst, "tpot")
    if scenario == "C":
        values = []
        for name in ("burst", "poisson", "constant"):
            profile = profiles.get(name)
            if not isinstance(profile, dict):
                raise ValueError(f"missing {name} profile")
            values.append(_throughput(profile))
        return PRIMARY_NAMES[scenario], _geomean(values)
    if scenario == "D":
        return PRIMARY_NAMES[scenario], _geomean(
            [1.0 / _p50(burst, "ttft"), 1.0 / _p50(burst, "tpot"), _throughput(burst)]
        )
    raise ValueError(f"unknown scenario: {scenario}")


def quality_metric(metrics: dict[str, Any]) -> tuple[str, float | None, dict[str, Any]]:
    quality = metrics.get("quality_check") or {}
    dataset = ((quality.get("datasets") or {}).get("mmlu_pro") or {})
    observed = dataset.get("observed_accuracy")
    value = float(observed) if isinstance(observed, (int, float)) else None
    return "quality/mmlu_pro_observed_accuracy", value, {
        "quality_pass": quality.get("pass"),
        "quality_ratio": dataset.get("ratio"),
        "quality_baseline_accuracy": dataset.get("baseline_accuracy"),
        "quality_tau": dataset.get("tau") or quality.get("tau"),
        "quality_sample_n": dataset.get("n") or dataset.get("evaluated_count"),
    }


def eval_mode(metrics: dict[str, Any]) -> dict[str, Any]:
    raw = metrics.get("eval_mode")
    if isinstance(raw, dict):
        name = str(raw.get("name") or "unknown").strip().lower()
        return {"name": name, **{k: v for k, v in raw.items() if k != "name"}}
    pipeline = metrics.get("eval_pipeline") or {}
    steps = pipeline.get("steps") if isinstance(pipeline, dict) else None
    if isinstance(steps, list) and "mock" in steps:
        return {"name": "mock"}
    return {"name": "full"}


def profile_health(metrics: dict[str, Any]) -> dict[str, Any]:
    profiles = metrics.get("profiles") or {}
    request_count = 0
    success_count = 0
    failure_count = 0
    max_failure_rate = 0.0
    for profile in profiles.values():
        if not isinstance(profile, dict):
            continue
        request_count += int(profile.get("request_count", 0) or 0)
        success_count += int(profile.get("success_count", 0) or 0)
        failure_count += int(profile.get("failure_count", 0) or 0)
        rate = profile.get("failure_rate")
        if isinstance(rate, (int, float)):
            max_failure_rate = max(max_failure_rate, float(rate))
    return {
        "speed_request_count": request_count,
        "speed_success_count": success_count,
        "speed_failure_count": failure_count,
        "speed_max_failure_rate": max_failure_rate,
    }


def eligibility_issues(
    *,
    result: dict[str, Any],
    mode: dict[str, Any],
    baseline_primary: float | None,
    quality: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    if not result.get("terminal"):
        issues.append("result is not terminal")
    if result.get("pending_arms"):
        issues.append("pending_arms is true")
    if mode.get("name") != "full":
        issues.append(f"eval_mode is {mode.get('name')!r}, not 'full'")
    if baseline_primary is None:
        issues.append("missing PyTorch baseline metric for speedup")
    if quality.get("quality_pass") is not True:
        issues.append("quality gate did not pass")
    qn = quality.get("quality_sample_n")
    if not isinstance(qn, (int, float)) or int(qn) < FULL_QUALITY_N:
        issues.append(f"quality sample n is {qn!r}, expected at least {FULL_QUALITY_N}")
    if quality.get("quality_ratio") is None:
        issues.append("missing quality ratio")
    return issues


def _load_metrics(path: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def _metrics_payload(data: dict[str, Any]) -> dict[str, Any]:
    baseline = data.get("baseline")
    if isinstance(baseline, dict) and isinstance(baseline.get("profiles"), dict):
        return baseline
    metrics = data.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("profiles"), dict):
        return metrics
    return data


def _baseline_primary_value(args: argparse.Namespace, scenario: str) -> float | None:
    if args.baseline_primary is not None:
        if args.baseline_primary <= 0:
            raise ValueError("--baseline-primary must be positive")
        return float(args.baseline_primary)
    if args.baseline_metrics_json:
        baseline_metrics = _metrics_payload(_load_metrics(args.baseline_metrics_json))
        _, value = primary_metric(baseline_metrics, scenario)
        return value
    return None


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    metrics = _metrics_payload(_load_metrics(args.metrics_json))
    scenario = infer_scenario(metrics, args.scenario)
    primary_name, primary_value = primary_metric(metrics, scenario)
    baseline_primary = _baseline_primary_value(args, scenario)
    speedup = (primary_value / baseline_primary) if baseline_primary else None
    test_name, test_value, quality = quality_metric(metrics)
    if test_value is None:
        test_name = primary_name
        test_value = primary_value
    mode = eval_mode(metrics)

    reported_primary_name = primary_name
    reported_primary_value = primary_value
    if speedup is not None:
        reported_primary_name = f"scenario/{scenario}/speedup_over_pytorch"
        reported_primary_value = speedup

    result: dict[str, Any] = {
        "terminal": args.terminal,
        "status": args.status,
        "pending_arms": args.pending_arms,
        "wandb_run_ids": args.wandb_run_id,
        "primary_metric": {"name": reported_primary_name, "value": reported_primary_value},
        "test_metric": {"name": test_name, "value": test_value},
        "scenario": scenario,
        "metrics_json": str(args.metrics_json),
        "raw_primary_metric": {"name": primary_name, "value": primary_value},
        "eval_mode": mode,
    }
    if baseline_primary is not None:
        result["baseline_primary_metric"] = {"name": primary_name, "value": baseline_primary}
        result["speedup_over_pytorch"] = speedup
    result.update(quality)
    result.update(profile_health(metrics))
    issues = eligibility_issues(result=result, mode=mode, baseline_primary=baseline_primary, quality=quality)
    result["terminal_eligible"] = not issues
    result["baseline_update_allowed"] = not issues
    result["terminal_eligibility_issues"] = issues
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", help="Path to InferenceBench metrics.json")
    parser.add_argument("--scenario", help="Scenario id: A, B, C, or D")
    parser.add_argument("--baseline-metrics-json", help="PyTorch baseline metrics.json for speedup reporting")
    parser.add_argument("--baseline-primary", type=float, help="Raw PyTorch baseline primary value for this scenario")
    parser.add_argument("--wandb-run-id", action="append", default=[], help="Optional W&B run id")
    parser.add_argument("--status", default="complete")
    parser.add_argument("--terminal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pending-arms", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--json-only", action="store_true", help="Print only the JSON payload")
    args = parser.parse_args()

    result = build_result(args)
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"))
    if args.json_only:
        print(payload)
    else:
        print(f"SENPAI-RESULT: {payload}")


if __name__ == "__main__":
    main()
