#!/usr/bin/env python3
"""Validate a full result and optionally promote its PR to review."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    from . import validate_result
except ImportError:  # pragma: no cover - script execution path
    import validate_result  # type: ignore


def result_marker(result: dict[str, Any]) -> str:
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"))
    return "SENPAI-RESULT: " + payload


def comment_body(result: dict[str, Any], launcher: str | None) -> str:
    lines = [
        result_marker(result),
        "",
        "## Terminal Result",
        "",
        f"- Scenario: {result.get('scenario')}",
        f"- Primary metric: `{(result.get('primary_metric') or {}).get('name')}` = {(result.get('primary_metric') or {}).get('value')}",
        f"- W&B runs: {', '.join(result.get('wandb_run_ids') or [])}",
        f"- Eval mode: `{(result.get('eval_mode') or {}).get('name')}`",
        f"- Quality pass: {result.get('quality_pass')} (ratio={result.get('quality_ratio')}, n={result.get('quality_sample_n')})",
        f"- Speed failures: {result.get('speed_failure_count')} (max failure rate={result.get('speed_max_failure_rate')})",
        "- Validation: `validation_pass=true`, `baseline_update_allowed=true`",
    ]
    if launcher:
        path = Path(launcher)
        lines.extend(["", "## Launcher", "", f"`{launcher}`"])
        if path.is_file():
            lines.extend(["", "```bash", path.read_text(encoding="utf-8").rstrip(), "```"])
    return "\n".join(lines) + "\n"


def _run(command: list[str], *, stdin: str | None = None) -> str:
    result = subprocess.run(command, input=stdin, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise SystemExit(f"{' '.join(command)} failed: {detail}")
    return result.stdout.strip()


def _post_comment(*, repo: str, pr: str, body: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(body)
        path = handle.name
    try:
        _run(["gh", "pr", "comment", pr, "--repo", repo, "--body-file", path])
    finally:
        Path(path).unlink(missing_ok=True)


def _mark_ready(*, repo: str, pr: str) -> None:
    is_draft = _run(["gh", "pr", "view", pr, "--repo", repo, "--json", "isDraft", "--jq", ".isDraft"])
    if is_draft == "true":
        _run(["gh", "pr", "ready", pr, "--repo", repo])


def _swap_labels(*, repo: str, pr: str, remove_label: str | None, add_label: str | None) -> None:
    if not remove_label and not add_label:
        return
    command = ["gh", "pr", "edit", pr, "--repo", repo]
    if remove_label:
        command.extend(["--remove-label", remove_label])
    if add_label:
        command.extend(["--add-label", add_label])
    _run(command)


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    validate_args = argparse.Namespace(
        metrics_json=args.metrics_json,
        scenario=args.scenario,
        baseline_metrics_json=args.baseline_metrics_json,
        baseline_primary=args.baseline_primary,
        wandb_run_id=args.wandb_run_id,
        status="complete",
        terminal=True,
        pending_arms=False,
        max_failure_rate=args.max_failure_rate,
        require_wandb=args.require_wandb,
        require_launcher=args.require_launcher,
        launcher=args.launcher,
        result_json=None,
    )
    result = validate_result.validate(validate_args)
    if not result["validation_pass"]:
        raise SystemExit("result is not terminal-valid: " + "; ".join(result["validation_issues"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", help="Full InferenceBench metrics.json")
    parser.add_argument("--scenario", required=True, help="Scenario id: A, B, C, or D")
    parser.add_argument("--baseline-metrics-json", required=True, help="Matching PyTorch baseline metrics.json")
    parser.add_argument("--baseline-primary", type=float)
    parser.add_argument("--wandb-run-id", action="append", default=[], help="W&B run id from full eval")
    parser.add_argument("--launcher", help="Final launcher path")
    parser.add_argument("--max-failure-rate", type=float, default=0.0)
    parser.add_argument("--require-wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-launcher", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-to-pr", action="store_true", help="Post terminal marker, mark ready, and swap labels")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--pr", help="Pull request number to promote")
    parser.add_argument("--ready", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--remove-label", default="status:wip")
    parser.add_argument("--add-label", default="status:review")
    parser.add_argument("--dry-run", action="store_true", help="Print the comment but do not call gh")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    result = finalize(args)
    if args.json_only:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return

    body = comment_body(result, args.launcher)
    print(body, end="")

    if not args.post_to_pr or args.dry_run:
        return
    if not args.repo or not args.pr:
        raise SystemExit("--post-to-pr requires --repo and --pr, or GITHUB_REPOSITORY plus --pr")
    _post_comment(repo=args.repo, pr=args.pr, body=body)
    if args.ready:
        _mark_ready(repo=args.repo, pr=args.pr)
    _swap_labels(repo=args.repo, pr=args.pr, remove_label=args.remove_label, add_label=args.add_label)


if __name__ == "__main__":
    main()
