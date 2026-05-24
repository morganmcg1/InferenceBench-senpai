#!/usr/bin/env python3
"""Create a pod-local InferenceBench task workspace for SENPAI students."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
from pathlib import Path


SCENARIOS = {
    "A": "inference_scenario_a_input_heavy",
    "B": "inference_scenario_b_output_heavy",
    "C": "inference_scenario_c_high_load",
    "D": "inference_scenario_d_general",
}


TEST_SERVER_WRAPPER = """#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${TASK_DIR}/.." && pwd)"
INFERENCE_EVAL_DIR="${INFERENCE_EVAL_DIR:-${WORKSPACE_DIR}/inference_eval}"

[ -f "${TASK_DIR}/eval_env.sh" ] && source "${TASK_DIR}/eval_env.sh"
exec "${INFERENCE_EVAL_DIR}/bin/launch_supervised_server.sh" "${TASK_DIR}/start_server.sh"
"""

CLEAN_EVAL_ARTIFACTS = """#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
rm -f \
  "${TASK_DIR}"/metrics*.json \
  "${TASK_DIR}"/requests_used_speed.jsonl \
  "${TASK_DIR}"/speed_generations.jsonl \
  "${TASK_DIR}"/quality_*_generations.jsonl \
  "${TASK_DIR}"/eval_generations.jsonl
"""

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def model_safe(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_") or "unknown_model"


def shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def copy_file(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_tree_contents(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def write_eval_env(task_dst: Path, workspace: Path, root: Path, scenario_folder: str) -> None:
    safe_model = model_safe(DEFAULT_MODEL)
    requests_file = workspace / "inference_eval" / "baselines" / "speed" / "torch" / scenario_folder / safe_model / "requests.jsonl"
    baseline_metrics = requests_file.with_name("baseline_metrics.json")
    quality_registry = workspace / "inference_eval" / "baselines" / "quality" / f"{safe_model}_torch.json"
    env = f"""#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
WORKSPACE_DIR="$(cd "${{TASK_DIR}}/.." && pwd)"

export PROBLEM_DIR="${{PROBLEM_DIR:-{shell_quote(root)}}}"
export INFERENCE_EVAL_DIR="${{INFERENCE_EVAL_DIR:-${{WORKSPACE_DIR}}/inference_eval}}"
export INFERENCE_BENCH_BASE_MODEL="${{INFERENCE_BENCH_BASE_MODEL:-{DEFAULT_MODEL}}}"
export INFERENCE_BENCH_SCENARIO="${{INFERENCE_BENCH_SCENARIO:-{scenario_folder}}}"
export INFERENCE_BENCH_DATASET_SEED="${{INFERENCE_BENCH_DATASET_SEED:-248}}"
export INFERENCE_BENCH_QUALITY_SEED="${{INFERENCE_BENCH_QUALITY_SEED:-${{INFERENCE_BENCH_DATASET_SEED}}}}"
export INFERENCE_BENCH_QUALITY_BASELINE_BACKEND="${{INFERENCE_BENCH_QUALITY_BASELINE_BACKEND:-torch}}"
export INFERENCE_BENCH_RUNTIME_CACHE_DIR="${{INFERENCE_BENCH_RUNTIME_CACHE_DIR:-/tmp/inferencebench-runtime-${{USER:-senpai}}}}"
export INFERENCE_BENCH_PYTORCH_BASELINE_METRICS="${{INFERENCE_BENCH_PYTORCH_BASELINE_METRICS:-{shell_quote(baseline_metrics)}}}"

if [ -f {shell_quote(requests_file)} ]; then
  export INFERENCE_BENCH_REQUESTS_FILE="${{INFERENCE_BENCH_REQUESTS_FILE:-{shell_quote(requests_file)}}}"
fi

if [ -f {shell_quote(quality_registry)} ]; then
  export INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY="${{INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY:-{shell_quote(quality_registry)}}}"
fi

if [ -f "${{PROBLEM_DIR}}/senpai/runtime_env.sh" ]; then
  source "${{PROBLEM_DIR}}/senpai/runtime_env.sh"
fi
"""
    path = task_dst / "eval_env.sh"
    path.write_text(env, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def evaluate_wrapper(scenario_folder: str) -> str:
    safe_model = model_safe(DEFAULT_MODEL)
    return f"""#!/usr/bin/env python3
import os
import sys
from pathlib import Path

workspace = Path(__file__).resolve().parents[1]
inference_dir = workspace / "inference_eval"
safe_model = {safe_model!r}
scenario = {scenario_folder!r}

os.environ.setdefault("INFERENCE_BENCH_BASE_MODEL", {DEFAULT_MODEL!r})
os.environ.setdefault("INFERENCE_BENCH_SCENARIO", scenario)
os.environ.setdefault("INFERENCE_BENCH_DATASET_SEED", "248")
os.environ.setdefault("INFERENCE_BENCH_QUALITY_SEED", os.environ["INFERENCE_BENCH_DATASET_SEED"])
os.environ.setdefault("INFERENCE_BENCH_QUALITY_BASELINE_BACKEND", "torch")
os.environ.setdefault("INFERENCE_BENCH_RUNTIME_CACHE_DIR", f"/tmp/inferencebench-runtime-{{os.environ.get('USER', 'senpai')}}")

requests_file = inference_dir / "baselines" / "speed" / "torch" / scenario / safe_model / "requests.jsonl"
if requests_file.is_file():
    os.environ.setdefault("INFERENCE_BENCH_REQUESTS_FILE", str(requests_file))
    os.environ.setdefault("INFERENCE_BENCH_PYTORCH_BASELINE_METRICS", str(requests_file.with_name("baseline_metrics.json")))

quality_registry = inference_dir / "baselines" / "quality" / f"{{safe_model}}_torch.json"
if quality_registry.is_file():
    os.environ.setdefault("INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY", str(quality_registry))

sys.path.insert(0, str(workspace))
from inference_eval.runner import build_parser, run_evaluation  # noqa: E402


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_evaluation(Path(__file__).resolve().parent, args)


if __name__ == "__main__":
    main()
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, help="A, B, C, D, or scenario folder name")
    parser.add_argument("--output", required=True, help="Workspace directory to create")
    parser.add_argument("--starting-point", choices=["default", "vllm_running", "bare"], default="vllm_running")
    parser.add_argument("--launcher", help="Optional launcher recipe to install as task/start_server.sh")
    parser.add_argument("--replace", action="store_true", help="Remove an existing workspace first")
    args = parser.parse_args()

    root = repo_root()
    scenario_key = args.scenario.strip().upper()
    scenario_folder = SCENARIOS.get(scenario_key, args.scenario.strip())
    task_src = root / "src" / "eval" / "tasks" / scenario_folder
    if not task_src.is_dir():
        raise SystemExit(f"scenario task not found: {task_src}")

    workspace = Path(args.output).expanduser().resolve()
    if workspace.exists() and args.replace:
        shutil.rmtree(workspace)
    task_dst = workspace / "task"
    task_dst.mkdir(parents=True, exist_ok=True)
    (task_dst / "agent").mkdir(parents=True, exist_ok=True)

    for name in ("mission.txt", "benchmark.txt", "scenario.json", "workspace_conventions.txt"):
        copy_file(task_src / name, task_dst / name)
    copy_tree_contents(root / "src" / "eval" / "tasks" / "_shared" / "task_context", task_dst)
    copy_tree_contents(task_src / "task_context", task_dst)

    if args.starting_point == "vllm_running":
        copy_file(root / "src" / "starting_points" / "vllm_running" / "start_server.sh", task_dst / "start_server.sh")
        copy_file(root / "src" / "starting_points" / "vllm_running" / "start_server.sh", task_dst / "start_server_original.sh")
    elif args.starting_point in {"default", "bare"}:
        copy_file(root / "src" / "eval" / "tasks" / "_shared" / "task_context" / "start_server.sh", task_dst / "start_server.sh")
    if args.launcher:
        copy_file(Path(args.launcher).expanduser().resolve(), task_dst / "start_server.sh")

    write_eval_env(task_dst, workspace, root, scenario_folder)
    (task_dst / "evaluate.py").write_text(evaluate_wrapper(scenario_folder), encoding="utf-8")
    (task_dst / "test_server.sh").write_text(TEST_SERVER_WRAPPER, encoding="utf-8")
    (task_dst / "clean_eval_artifacts.sh").write_text(CLEAN_EVAL_ARTIFACTS, encoding="utf-8")
    for script in task_dst.glob("*.sh"):
        script.chmod(script.stat().st_mode | 0o111)

    inference_dst = workspace / "inference_eval"
    if inference_dst.exists():
        shutil.rmtree(inference_dst)
    inference_dst.mkdir(parents=True)
    for name in ("__init__.py", "runner.py", "quality_gate.py", "cache_samples.py"):
        copy_file(root / "src" / "eval" / "inference" / name, inference_dst / name)
    copy_tree_contents(root / "src" / "eval" / "inference" / "bin", inference_dst / "bin")

    baselines_src = root / "src" / "eval" / "inference" / "baselines"
    baselines_dst = inference_dst / "baselines"
    if baselines_src.exists():
        try:
            os.symlink(baselines_src, baselines_dst, target_is_directory=True)
        except OSError:
            shutil.copytree(baselines_src, baselines_dst)

    print(f"[workspace] created {workspace}")
    print(f"[workspace] task dir: {task_dst}")
    print(f"[workspace] env: source {task_dst / 'eval_env.sh'}")
    print(f"[workspace] baseline: $INFERENCE_BENCH_PYTORCH_BASELINE_METRICS")
    print(f"[workspace] run: cd {task_dst} && ./test_server.sh > agent/server.log 2>&1 &")
    print(f"[workspace] clean: cd {task_dst} && ./clean_eval_artifacts.sh")
    print(f"[workspace] eval: cd {task_dst} && python evaluate.py --quick --json-output-file metrics_quick.json")


if __name__ == "__main__":
    main()
