#!/usr/bin/env python3
"""Validate that an InferenceBench result is eligible for BASELINE.md updates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from . import preflight, summarize_metrics
except ImportError:  # pragma: no cover - script execution path
    import preflight  # type: ignore
    import summarize_metrics  # type: ignore


EXPECTED_PROFILES = {
    "A": ["burst"],
    "B": ["burst"],
    "C": ["burst", "poisson", "constant"],
    "D": ["burst"],
}


def _load_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def _result_payload(path_or_json: str | None) -> dict[str, Any] | None:
    if not path_or_json:
        return None
    candidate = Path(path_or_json)
    try:
        is_file = candidate.is_file()
    except OSError:
        is_file = False
    raw = candidate.read_text(encoding="utf-8") if is_file else path_or_json
    prefix = "SENPAI-RESULT:"
    if prefix in raw:
        raw = raw.split(prefix, 1)[1].strip()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--result-json must be a JSON object or a file containing one")
    return data


def _close(left: Any, right: Any) -> bool:
    return isinstance(left, (int, float)) and isinstance(right, (int, float)) and math.isclose(
        float(left),
        float(right),
        rel_tol=1e-6,
        abs_tol=1e-12,
    )


def _quality_note(metrics: dict[str, Any]) -> str:
    quality = metrics.get("quality_check") or {}
    note = quality.get("note") or ""
    error = quality.get("error") or ""
    return " ".join(str(part) for part in (note, error) if part)


def _validate_profiles(
    *,
    metrics: dict[str, Any],
    scenario: str,
    max_failure_rate: float,
) -> list[str]:
    issues: list[str] = []
    profiles = metrics.get("profiles") or {}
    if not isinstance(profiles, dict):
        return ["metrics.profiles is missing or not an object"]

    scenario_folder = preflight.SCENARIOS[scenario]
    expected_n = preflight.scenario_request_count(preflight.repo_root(), scenario_folder)
    for name in EXPECTED_PROFILES[scenario]:
        profile = profiles.get(name)
        if not isinstance(profile, dict):
            issues.append(f"missing speed profile {name!r}")
            continue
        request_count = int(profile.get("request_count", 0) or 0)
        success_count = int(profile.get("success_count", 0) or 0)
        failure_count = int(profile.get("failure_count", 0) or 0)
        failure_rate = float(profile.get("failure_rate", 0.0) or 0.0)
        if request_count < expected_n:
            issues.append(f"profile {name!r} has request_count={request_count}, expected at least {expected_n}")
        if success_count < request_count:
            issues.append(f"profile {name!r} has success_count={success_count} < request_count={request_count}")
        if failure_count:
            issues.append(f"profile {name!r} has failure_count={failure_count}")
        if failure_rate > max_failure_rate:
            issues.append(f"profile {name!r} has failure_rate={failure_rate} > {max_failure_rate}")
    return issues


def validate(args: argparse.Namespace) -> dict[str, Any]:
    metrics = summarize_metrics._metrics_payload(_load_json(args.metrics_json))  # pylint: disable=protected-access
    summary_args = argparse.Namespace(
        metrics_json=args.metrics_json,
        scenario=args.scenario,
        baseline_metrics_json=args.baseline_metrics_json,
        baseline_primary=args.baseline_primary,
        wandb_run_id=args.wandb_run_id or [],
        status=args.status,
        terminal=args.terminal,
        pending_arms=args.pending_arms,
    )
    result = summarize_metrics.build_result(summary_args)
    scenario = str(result["scenario"])
    issues = list(result.get("terminal_eligibility_issues") or [])

    if metrics.get("error"):
        issues.append(f"metrics contains evaluator error: {metrics.get('error')}")
    if "skipped" in _quality_note(metrics).lower():
        issues.append("quality check appears to have been skipped")
    issues.extend(_validate_profiles(metrics=metrics, scenario=scenario, max_failure_rate=args.max_failure_rate))

    if args.require_wandb and not result.get("wandb_run_ids"):
        issues.append("missing W&B run id")
    if args.require_launcher:
        if not args.launcher:
            issues.append("missing launcher path")
        else:
            launcher = Path(args.launcher)
            if not launcher.is_file() or launcher.stat().st_size == 0:
                issues.append(f"launcher path is missing or empty: {launcher}")

    payload = _result_payload(args.result_json)
    if payload is not None:
        if args.terminal and payload.get("terminal") is not True:
            issues.append("SENPAI-RESULT terminal is not true")
        if payload.get("pending_arms"):
            issues.append("SENPAI-RESULT pending_arms is true")
        if payload.get("scenario") and str(payload.get("scenario")).upper() != scenario:
            issues.append(f"SENPAI-RESULT scenario {payload.get('scenario')!r} does not match {scenario}")
        reported = payload.get("primary_metric") or {}
        expected = result.get("primary_metric") or {}
        if reported.get("name") != expected.get("name"):
            issues.append(
                f"SENPAI-RESULT primary_metric.name {reported.get('name')!r} "
                f"does not match computed {expected.get('name')!r}"
            )
        if not _close(reported.get("value"), expected.get("value")):
            issues.append(
                f"SENPAI-RESULT primary_metric.value {reported.get('value')!r} "
                f"does not match computed {expected.get('value')!r}"
            )
        if payload.get("baseline_update_allowed") is False:
            issues.append("SENPAI-RESULT explicitly sets baseline_update_allowed=false")

    result["validation_issues"] = sorted(set(str(issue) for issue in issues))
    result["validation_pass"] = not result["validation_issues"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", help="Path to InferenceBench metrics.json")
    parser.add_argument("--scenario", help="Scenario id: A, B, C, or D")
    parser.add_argument("--baseline-metrics-json", help="Matching PyTorch baseline metrics.json")
    parser.add_argument("--baseline-primary", type=float, help="Raw PyTorch baseline primary value")
    parser.add_argument("--wandb-run-id", action="append", default=[], help="W&B run id to require/report")
    parser.add_argument("--launcher", help="Launcher recipe path to require/report")
    parser.add_argument("--result-json", help="Optional SENPAI-RESULT JSON payload or file")
    parser.add_argument("--status", default="complete")
    parser.add_argument("--terminal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pending-arms", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-failure-rate", type=float, default=0.0)
    parser.add_argument("--require-wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-launcher", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    result = validate(args)
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"))
    if args.json_only:
        print(payload)
    else:
        print("SENPAI-VALIDATION: " + payload)
        if result["validation_issues"]:
            for issue in result["validation_issues"]:
                print(f"[FAIL] {issue}")
        else:
            print("[PASS] terminal result is eligible for BASELINE.md update")
    raise SystemExit(0 if result["validation_pass"] else 1)


if __name__ == "__main__":
    main()
