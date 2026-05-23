from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from senpai import create_task_workspace, materialize_requests, preflight


def test_tokenized_length_handles_batchencoding_like_dict() -> None:
    assert materialize_requests.tokenized_length({"input_ids": [[1, 2, 3, 4]]}) == 4
    assert preflight.tokenized_length({"input_ids": [1, 2, 3]}) == 3


def test_preflight_finds_nested_speed_baseline(tmp_path: Path) -> None:
    safe_model = preflight.model_safe("org/model")
    scenario = "inference_scenario_a_input_heavy"
    metrics = tmp_path / "src/eval/inference/baselines/speed/torch" / scenario / safe_model / "baseline_metrics.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text('{"baseline": {"profiles": {"burst": {}}}}', encoding="utf-8")

    found = preflight.find_first(preflight.speed_baseline_candidates(tmp_path, "torch", scenario, safe_model))

    assert found == metrics


def test_create_task_workspace_copies_task_files(tmp_path: Path) -> None:
    out = tmp_path / "workspace"
    # Exercise the script through its file operations without requiring a GPU.
    argv = sys.argv
    try:
        sys.argv = [
            "create_task_workspace.py",
            "--scenario",
            "A",
            "--output",
            str(out),
            "--starting-point",
            "bare",
        ]
        create_task_workspace.main()
    finally:
        sys.argv = argv

    assert (out / "task" / "evaluate.py").is_file()
    assert (out / "task" / "scenario.json").is_file()
    assert (out / "inference_eval" / "runner.py").is_file()
