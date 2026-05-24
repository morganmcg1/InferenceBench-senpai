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


def test_preflight_rejects_partial_speed_baseline(tmp_path: Path) -> None:
    safe_model = preflight.model_safe("org/model")
    scenario = "inference_scenario_a_input_heavy"
    scenario_dir = tmp_path / "src/eval/tasks" / scenario
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "scenario.json").write_text('{"num_requests": 4}', encoding="utf-8")
    metrics = tmp_path / "src/eval/inference/baselines/speed/torch" / scenario / safe_model / "baseline_metrics.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(
        '{"baseline": {"request_count": 4, "success_count": 3, "profiles": {"burst": {"success_count": 3}}}}',
        encoding="utf-8",
    )
    requests = metrics.parent / "requests.jsonl"
    requests.write_text("\n".join(["{}"] * 4) + "\n", encoding="utf-8")

    checks, requests_ready = preflight.check_speed_assets(tmp_path, ["A"], safe_model, "torch")

    assert requests_ready is True
    baseline_check = next(c for c in checks if c.name == "speed_baseline_A")
    assert baseline_check.status == "fail"
    assert "success_count=3 < request_count=4" in baseline_check.detail


def test_preflight_does_not_require_tokenizer_when_requests_ready(monkeypatch) -> None:
    from src.eval.inference import runner

    monkeypatch.setattr(runner, "_get_tokenizer", lambda _model_id: None)

    check = preflight.check_tokenizer("org/model", request_files_ready=True)

    assert check.status == "pass"
    assert "pre-materialized request files are present" in check.detail


def test_preflight_fails_without_tokenizer_or_requests(monkeypatch) -> None:
    from src.eval.inference import runner

    monkeypatch.setattr(runner, "_get_tokenizer", lambda _model_id: None)

    check = preflight.check_tokenizer("org/model", request_files_ready=False)

    assert check.status == "fail"
    assert "deterministic request files are incomplete" in check.detail


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
