import argparse
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from senpai import (
    create_task_workspace,
    finalize_result,
    gpu_slot,
    materialize_requests,
    preflight,
    summarize_metrics,
    validate_result,
)


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


def test_runtime_import_env_prepends_nvidia_wheel_libs(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "nvidia_wheel_library_paths", lambda: ["/site/nvidia/cuda_runtime/lib"])
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")

    env = preflight.runtime_import_env()

    assert env["LD_LIBRARY_PATH"] == "/site/nvidia/cuda_runtime/lib:/existing"


def test_runtime_env_disables_implicit_flashinfer_for_shakedown() -> None:
    text = Path("senpai/runtime_env.sh").read_text(encoding="utf-8")

    assert 'INFERENCE_BENCH_MAX_MODEL_LEN="${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}"' in text
    assert 'VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"' in text
    assert 'VLLM_DISABLE_FLASHINFER_PREFILL="${VLLM_DISABLE_FLASHINFER_PREFILL:-1}"' in text
    assert 'PIP_REQUIRE_VIRTUALENV="${PIP_REQUIRE_VIRTUALENV:-true}"' in text


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
    assert (out / "task" / "eval_env.sh").is_file()
    assert (out / "task" / "clean_eval_artifacts.sh").is_file()
    assert (out / "task" / "scenario.json").is_file()
    assert (out / "inference_eval" / "runner.py").is_file()

    env_text = (out / "task" / "eval_env.sh").read_text(encoding="utf-8")
    assert "INFERENCE_BENCH_BASE_MODEL" in env_text
    assert "INFERENCE_BENCH_SCENARIO" in env_text
    assert "INFERENCE_BENCH_PYTORCH_BASELINE_METRICS" in env_text


def test_summarize_metrics_reads_wrapped_baseline(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        '{"scenario":"A","profiles":{"burst":{"ttft":{"p50":2.0},"tpot":{"p50":1.0},'
        '"request_throughput_req_per_s":1.0}},"quality_check":{"pass":true}}',
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline_metrics.json"
    baseline.write_text(
        '{"baseline":{"profiles":{"burst":{"ttft":{"p50":4.0},"tpot":{"p50":1.0},'
        '"request_throughput_req_per_s":1.0}}}}',
        encoding="utf-8",
    )

    class Args:
        metrics_json = str(metrics)
        scenario = "A"
        baseline_metrics_json = str(baseline)
        baseline_primary = None
        wandb_run_id = []
        status = "complete"
        terminal = True
        pending_arms = False

    result = summarize_metrics.build_result(Args())

    assert result["primary_metric"]["name"] == "scenario/A/speedup_over_pytorch"
    assert result["primary_metric"]["value"] == 2.0


def test_summarize_marks_quick_results_ineligible(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics_quick.json"
    metrics.write_text(
        '{"scenario":"A","profiles":{"burst":{"request_count":24,"success_count":24,'
        '"failure_count":0,"failure_rate":0.0,"ttft":{"p50":2.0},"tpot":{"p50":1.0},'
        '"request_throughput_req_per_s":1.0}},"quality_check":{"pass":true,'
        '"datasets":{"mmlu_pro":{"observed_accuracy":0.5,"baseline_accuracy":0.5,'
        '"ratio":1.0,"n":16}}},"eval_mode":{"name":"quick","request_limit":24,"quality_n":16}}',
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline_metrics.json"
    baseline.write_text(
        '{"baseline":{"profiles":{"burst":{"ttft":{"p50":4.0},"tpot":{"p50":1.0},'
        '"request_throughput_req_per_s":1.0}}}}',
        encoding="utf-8",
    )

    args = argparse.Namespace(
        metrics_json=str(metrics),
        scenario="A",
        baseline_metrics_json=str(baseline),
        baseline_primary=None,
        wandb_run_id=["abc123"],
        status="complete",
        terminal=True,
        pending_arms=False,
    )

    result = summarize_metrics.build_result(args)

    assert result["eval_mode"]["name"] == "quick"
    assert result["terminal_eligible"] is False
    assert result["baseline_update_allowed"] is False
    assert result["result_kind"] == "research_signal"
    assert result["quality_evidence"] == "screening_only"
    assert any("eval_mode" in issue for issue in result["terminal_eligibility_issues"])


def test_validate_result_accepts_full_terminal_metrics(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics_full.json"
    metrics.write_text(
        '{"scenario":"A","profiles":{"burst":{"request_count":128,"success_count":128,'
        '"failure_count":0,"failure_rate":0.0,"ttft":{"p50":2.0},"tpot":{"p50":1.0},'
        '"request_throughput_req_per_s":1.0}},"quality_check":{"pass":true,'
        '"datasets":{"mmlu_pro":{"observed_accuracy":0.5,"baseline_accuracy":0.5,'
        '"ratio":1.0,"n":500}}}}',
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline_metrics.json"
    baseline.write_text(
        '{"baseline":{"profiles":{"burst":{"ttft":{"p50":4.0},"tpot":{"p50":1.0},'
        '"request_throughput_req_per_s":1.0}}}}',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        metrics_json=str(metrics),
        scenario="A",
        baseline_metrics_json=str(baseline),
        baseline_primary=None,
        wandb_run_id=["abc123"],
        status="complete",
        terminal=True,
        pending_arms=False,
        max_failure_rate=0.0,
        require_wandb=True,
        require_launcher=False,
        launcher=None,
        result_json=None,
    )

    result = validate_result.validate(args)

    assert result["validation_pass"] is True
    assert result["baseline_update_allowed"] is True
    assert result["result_kind"] == "terminal_candidate"
    assert result["quality_evidence"] == "full_quality_gate"


def test_finalize_result_builds_terminal_comment(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics_full.json"
    metrics.write_text(
        '{"scenario":"A","profiles":{"burst":{"request_count":128,"success_count":128,'
        '"failure_count":0,"failure_rate":0.0,"ttft":{"p50":2.0},"tpot":{"p50":1.0},'
        '"request_throughput_req_per_s":1.0}},"quality_check":{"pass":true,'
        '"datasets":{"mmlu_pro":{"observed_accuracy":0.5,"baseline_accuracy":0.5,'
        '"ratio":1.0,"n":500}}}}',
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline_metrics.json"
    baseline.write_text(
        '{"baseline":{"profiles":{"burst":{"ttft":{"p50":4.0},"tpot":{"p50":1.0},'
        '"request_throughput_req_per_s":1.0}}}}',
        encoding="utf-8",
    )
    launcher = tmp_path / "start_server.sh"
    launcher.write_text("#!/usr/bin/env bash\nexec python -m vllm.entrypoints.openai.api_server\n", encoding="utf-8")
    args = argparse.Namespace(
        metrics_json=str(metrics),
        scenario="A",
        baseline_metrics_json=str(baseline),
        baseline_primary=None,
        wandb_run_id=["abc123"],
        max_failure_rate=0.0,
        require_wandb=True,
        require_launcher=True,
        launcher=str(launcher),
    )

    result = finalize_result.finalize(args)
    body = finalize_result.comment_body(result, str(launcher))

    assert result["validation_pass"] is True
    assert body.startswith("SENPAI-RESULT: ")
    assert '"baseline_update_allowed":true' in body
    assert "status:review" not in body


def test_validate_result_rejects_quick_metrics(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics_quick.json"
    metrics.write_text(
        '{"scenario":"A","profiles":{"burst":{"request_count":24,"success_count":24,'
        '"failure_count":0,"failure_rate":0.0,"ttft":{"p50":2.0},"tpot":{"p50":1.0},'
        '"request_throughput_req_per_s":1.0}},"quality_check":{"pass":true,'
        '"datasets":{"mmlu_pro":{"observed_accuracy":0.5,"baseline_accuracy":0.5,'
        '"ratio":1.0,"n":16}}},"eval_mode":{"name":"quick","request_limit":24,"quality_n":16}}',
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline_metrics.json"
    baseline.write_text(
        '{"baseline":{"profiles":{"burst":{"ttft":{"p50":4.0},"tpot":{"p50":1.0},'
        '"request_throughput_req_per_s":1.0}}}}',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        metrics_json=str(metrics),
        scenario="A",
        baseline_metrics_json=str(baseline),
        baseline_primary=None,
        wandb_run_id=["abc123"],
        status="complete",
        terminal=True,
        pending_arms=False,
        max_failure_rate=0.0,
        require_wandb=True,
        require_launcher=False,
        launcher=None,
        result_json=None,
    )

    result = validate_result.validate(args)

    assert result["validation_pass"] is False
    assert any("eval_mode" in issue for issue in result["validation_issues"])
    assert any("request_count=24" in issue for issue in result["validation_issues"])


def test_gpu_slot_acquire_release(tmp_path: Path) -> None:
    path = tmp_path / "slot.json"

    class Args:
        owner = "student-a"
        pr = "123"
        scenario = "C"
        purpose = "test"
        ttl_s = 60
        wait = False
        wait_timeout_s = None
        poll_s = 1

    lease = gpu_slot.acquire(path, Args())
    assert lease["owner"] == "student-a"
    assert gpu_slot.read_lease(path)["pr"] == "123"

    assert gpu_slot.heartbeat(path, "student-a", pr="123", ttl_s=120) is True
    assert gpu_slot.release(path, "student-a", pr="123") is True
    assert gpu_slot.read_lease(path) is None


def test_gpu_slot_quick_mode_caps_lease_policy(tmp_path: Path) -> None:
    path = tmp_path / "slot.json"

    class Args:
        owner = "student-a"
        pr = "123"
        scenario = "B"
        purpose = "quick"
        mode = "quick"
        ttl_s = 1800
        max_runtime_s = None
        min_remaining_s = 0
        wait = False
        wait_timeout_s = None
        poll_s = 1
        ignore_active_gpu = False
        replace_existing = False

    lease = gpu_slot.acquire(path, Args())

    assert lease["mode"] == "quick"
    assert lease["ttl_s"] == 900
    assert lease["min_remaining_s"] == 900
    assert lease["max_runtime_s"] == 900


def test_gpu_slot_run_enforces_runtime_limit(tmp_path: Path) -> None:
    path = tmp_path / "slot.json"
    args = argparse.Namespace(
        slot_file=str(path),
        owner="student-a",
        pr="123",
        scenario="B",
        purpose="test",
        mode="custom",
        ttl_s=60,
        max_runtime_s=1,
        min_remaining_s=0,
        wait=False,
        wait_timeout_s=None,
        poll_s=1,
        ignore_active_gpu=True,
        replace_existing=False,
        deadline_utc=None,
        deadline_epoch=None,
        command=[sys.executable, "-c", "import time; time.sleep(30)"],
    )

    assert gpu_slot.cmd_run(args) == gpu_slot.RUNTIME_LIMIT_EXIT
    assert gpu_slot.read_lease(path) is None


def test_gpu_slot_same_owner_pr_does_not_replace_without_flag(tmp_path: Path) -> None:
    path = tmp_path / "slot.json"

    class Args:
        owner = "student-a"
        pr = "123"
        scenario = "C"
        purpose = "test"
        ttl_s = 60
        wait = False
        wait_timeout_s = None
        poll_s = 1
        ignore_active_gpu = False
        replace_existing = False

    first = gpu_slot.acquire(path, Args())

    with pytest.raises(SystemExit) as exc:
        gpu_slot.acquire(path, Args())

    assert "GPU slot held" in str(exc.value)
    assert gpu_slot.read_lease(path)["lease_id"] == first["lease_id"]


def test_gpu_slot_old_runner_cannot_release_replaced_lease(tmp_path: Path) -> None:
    path = tmp_path / "slot.json"

    class Args:
        owner = "student-a"
        pr = "123"
        scenario = "C"
        purpose = "test"
        ttl_s = 60
        wait = False
        wait_timeout_s = None
        poll_s = 1
        ignore_active_gpu = False
        replace_existing = False

    class ReplaceArgs(Args):
        replace_existing = True

    old = gpu_slot.acquire(path, Args())
    new = gpu_slot.acquire(path, ReplaceArgs())

    assert gpu_slot.release(path, "student-a", pr="123", lease_id=old["lease_id"]) is False
    assert gpu_slot.read_lease(path)["lease_id"] == new["lease_id"]
    assert gpu_slot.release(path, "student-a", pr="123", lease_id=new["lease_id"]) is True


def test_gpu_slot_refuses_unleased_active_gpu(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "slot.json"

    class Args:
        owner = "student-a"
        pr = "123"
        scenario = "C"
        purpose = "test"
        ttl_s = 60
        wait = False
        wait_timeout_s = None
        poll_s = 1
        ignore_active_gpu = False
        replace_existing = False

    monkeypatch.setattr(
        gpu_slot,
        "active_gpu_processes",
        lambda: [{"pid": 4242, "process_name": "VLLM::EngineCore", "used_memory_mb": 90000}],
    )

    with pytest.raises(SystemExit) as exc:
        gpu_slot.acquire(path, Args())

    assert "GPU slot has no lease but GPU is busy" in str(exc.value)
    assert gpu_slot.read_lease(path) is None


def test_gpu_slot_refuses_work_too_close_to_deadline(tmp_path: Path) -> None:
    path = tmp_path / "slot.json"

    class Args:
        owner = "student-a"
        pr = "123"
        scenario = "C"
        purpose = "test"
        ttl_s = 60
        wait = False
        wait_timeout_s = None
        poll_s = 1
        ignore_active_gpu = False
        replace_existing = False
        deadline_utc = str(gpu_slot.now() + 10)
        deadline_epoch = None
        min_remaining_s = 60

    with pytest.raises(SystemExit) as exc:
        gpu_slot.acquire(path, Args())

    assert "less than required min_remaining_s=60" in str(exc.value)
    assert gpu_slot.read_lease(path) is None
