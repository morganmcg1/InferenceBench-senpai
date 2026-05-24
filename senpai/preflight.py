#!/usr/bin/env python3
"""Check whether an InferenceBench SENPAI launch can produce paper-grade scores."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
SCENARIOS = {
    "A": "inference_scenario_a_input_heavy",
    "B": "inference_scenario_b_output_heavy",
    "C": "inference_scenario_c_high_load",
    "D": "inference_scenario_d_general",
}


@dataclass
class Check:
    name: str
    status: str
    detail: str
    data: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
        }
        if self.data:
            payload["data"] = self.data
        return payload


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def model_safe(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_") or "unknown_model"


def selected_scenarios(raw: str) -> list[str]:
    if raw.lower() == "all":
        return list(SCENARIOS)
    out: list[str] = []
    for item in raw.split(","):
        key = item.strip().upper()
        if key not in SCENARIOS:
            raise SystemExit(f"unknown scenario {item!r}; use A,B,C,D,all")
        out.append(key)
    return out


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def maybe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def scenario_request_count(root: Path, scenario_folder: str) -> int:
    cfg = maybe_load_json(root / "src" / "eval" / "tasks" / scenario_folder / "scenario.json") or {}
    profiles = cfg.get("profiles")
    if isinstance(profiles, list):
        return max(int(p.get("num_requests", 0) or 0) for p in profiles if isinstance(p, dict))
    return int(cfg.get("num_requests", 0) or 0)


def speed_baseline_candidates(root: Path, backend: str, scenario_folder: str, safe_model: str) -> list[Path]:
    base = root / "src" / "eval" / "inference" / "baselines" / "speed" / backend
    return [
        base / scenario_folder / safe_model / "baseline_metrics.json",
        base / scenario_folder / "baseline_metrics.json",
    ]


def request_file_candidates(root: Path, backend: str, scenario_folder: str, safe_model: str) -> list[Path]:
    base = root / "src" / "eval" / "inference" / "baselines" / "speed" / backend
    return [
        base / scenario_folder / safe_model / "requests.jsonl",
        base / scenario_folder / "requests.jsonl",
    ]


def find_first(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def quality_registry_candidates(root: Path, backend: str, safe_model: str) -> list[Path]:
    backend_safe = model_safe(backend)
    if backend_safe == "vllm":
        legacy = root / "src" / "eval" / "inference" / "baselines" / "quality" / f"{safe_model}.json"
    else:
        legacy = root / "src" / "eval" / "inference" / "baselines" / "quality" / f"{safe_model}_{backend_safe}.json"
    return [
        legacy,
        root
        / "src"
        / "eval"
        / "inference"
        / "baselines"
        / "results"
        / safe_model
        / f"quality_{backend_safe}.json",
    ]


def check_hardware(expected_gpu: str, require_expected: bool) -> Check:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except Exception as exc:
        status = "fail" if require_expected else "warn"
        return Check("hardware", status, f"nvidia-smi unavailable: {exc}")

    if result.returncode != 0:
        status = "fail" if require_expected else "warn"
        return Check("hardware", status, result.stderr.strip() or "nvidia-smi returned non-zero")

    gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    expected_ok = any(expected_gpu.lower() in line.lower() for line in gpus)
    if expected_ok:
        return Check("hardware", "pass", f"found expected GPU substring {expected_gpu!r}", {"gpus": gpus})
    status = "fail" if require_expected else "warn"
    return Check(
        "hardware",
        status,
        f"expected GPU substring {expected_gpu!r} not found",
        {"gpus": gpus},
    )


def check_wandb(require_wandb: bool) -> Check:
    has_key = bool(os.environ.get("WANDB_API_KEY"))
    try:
        __import__("wandb")
        has_module = True
    except Exception:
        has_module = False
    if has_key and has_module:
        return Check("wandb", "pass", "WANDB_API_KEY is set and wandb imports")
    status = "fail" if require_wandb else "warn"
    missing = []
    if not has_key:
        missing.append("WANDB_API_KEY")
    if not has_module:
        missing.append("wandb module")
    return Check("wandb", status, "missing " + ", ".join(missing))


def check_runtime_imports(require_vllm: bool) -> Check:
    code = r'''
import json
import sys

report = {}
try:
    import torch
    report["torch_version"] = getattr(torch, "__version__", "unknown")
    report["torch_cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
    report["cuda_available"] = bool(torch.cuda.is_available())
except Exception as exc:
    print(json.dumps({"error": f"torch import failed: {exc}"}))
    sys.exit(2)

try:
    import vllm
    report["vllm_version"] = getattr(vllm, "__version__", "unknown")
except Exception as exc:
    report["vllm_error"] = str(exc)
    sys.exit(3)

print(json.dumps(report, sort_keys=True))
'''
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            check=False,
            timeout=45,
        )
    except Exception as exc:
        status = "fail" if require_vllm else "warn"
        return Check("python_runtime", status, f"runtime import check failed to run: {exc}")

    stdout = result.stdout.strip()
    data = maybe_json(stdout)
    if result.returncode == 0:
        return Check("python_runtime", "pass", "torch and vLLM import cleanly", data)
    status = "fail" if require_vllm else "warn"
    detail = stdout or result.stderr.strip() or f"runtime import check exited {result.returncode}"
    return Check("python_runtime", status, detail, data)


def maybe_json(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def check_tokenizer(model_id: str, request_files_ready: bool) -> Check:
    if request_files_ready:
        return Check(
            "tokenizer_request_sampling",
            "pass",
            "pre-materialized request files are present; tokenizer download is not required for request sampling",
        )

    root = repo_root()
    sys.path.insert(0, str(root))
    try:
        from src.eval.inference import runner  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        return Check("tokenizer_request_sampling", "warn", f"could not import runner: {exc}")

    tok = runner._get_tokenizer(model_id)  # pylint: disable=protected-access
    if tok is None:
        return Check(
            "tokenizer_request_sampling",
            "fail",
            "tokenizer unavailable locally and deterministic request files are incomplete; "
            "set INFERENCE_BENCH_ALLOW_HF_DOWNLOAD=1 or pre-materialize request files",
        )

    messages = [{"role": "user", "content": "hello " * 256}]
    try:
        raw = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    except Exception as exc:
        return Check("tokenizer_request_sampling", "warn", f"apply_chat_template failed: {exc}")

    expected = tokenized_length(raw)
    observed = runner._count_chat_tokens(messages, tok)  # pylint: disable=protected-access
    if observed == expected and observed > 16:
        return Check("tokenizer_request_sampling", "pass", f"runner token count sane ({observed} tokens)")

    status = "warn" if request_files_ready else "fail"
    detail = (
        f"runner token count looks wrong: observed={observed}, expected={expected}. "
        "Use pre-materialized requests from senpai/materialize_requests.py before running eval."
    )
    return Check("tokenizer_request_sampling", status, detail)


def tokenized_length(tokenized: Any) -> int:
    if isinstance(tokenized, dict) and "input_ids" in tokenized:
        ids = tokenized["input_ids"]
    elif hasattr(tokenized, "keys") and "input_ids" in tokenized.keys():
        ids = tokenized["input_ids"]
    else:
        ids = tokenized
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        return len(ids[0])
    return len(ids)


def check_speed_assets(
    root: Path,
    scenarios: list[str],
    safe_model: str,
    baseline_backend: str,
) -> tuple[list[Check], bool]:
    checks: list[Check] = []
    all_requests_ready = True
    for scenario in scenarios:
        folder = SCENARIOS[scenario]
        expected_n = scenario_request_count(root, folder)
        baseline = find_first(speed_baseline_candidates(root, baseline_backend, folder, safe_model))
        if baseline is None:
            checks.append(
                Check(
                    f"speed_baseline_{scenario}",
                    "fail",
                    f"missing {baseline_backend} baseline_metrics.json for {folder}",
                )
            )
        else:
            data = maybe_load_json(baseline) or {}
            baseline_payload = data.get("baseline") if isinstance(data.get("baseline"), dict) else data
            request_count = int(baseline_payload.get("request_count", 0) or 0)
            success_count = int(baseline_payload.get("success_count", 0) or 0)
            profiles = baseline_payload.get("profiles") if isinstance(baseline_payload, dict) else {}
            if request_count < expected_n:
                checks.append(
                    Check(
                        f"speed_baseline_{scenario}",
                        "fail",
                        f"{baseline.relative_to(root)} has request_count={request_count}, expected at least {expected_n}",
                    )
                )
            elif success_count < request_count:
                checks.append(
                    Check(
                        f"speed_baseline_{scenario}",
                        "fail",
                        f"{baseline.relative_to(root)} has success_count={success_count} < request_count={request_count}",
                    )
                )
            elif not isinstance(profiles, dict) or not profiles:
                checks.append(
                    Check(
                        f"speed_baseline_{scenario}",
                        "fail",
                        f"{baseline.relative_to(root)} has no profile metrics",
                    )
                )
            else:
                checks.append(
                    Check(
                        f"speed_baseline_{scenario}",
                        "pass",
                        f"found {baseline.relative_to(root)} with {success_count}/{request_count} successful baseline requests",
                        {"has_baseline_wrapper": "baseline" in data},
                    )
                )

        req = find_first(request_file_candidates(root, baseline_backend, folder, safe_model))
        if req is None:
            all_requests_ready = False
            checks.append(
                Check(
                    f"requests_{scenario}",
                    "fail",
                    f"missing deterministic requests.jsonl for {folder}",
                )
            )
        else:
            n = count_jsonl(req)
            status = "pass" if n >= expected_n else "warn"
            if status != "pass":
                all_requests_ready = False
            checks.append(
                Check(
                    f"requests_{scenario}",
                    status,
                    f"found {req.relative_to(root)} with {n} rows (expected at least {expected_n})",
                )
            )
    return checks, all_requests_ready


def check_quality_assets(root: Path, safe_model: str, backend: str, seed: int, n: int) -> list[Check]:
    checks: list[Check] = []
    sample_path = (
        root
        / "src"
        / "eval"
        / "inference"
        / "baselines"
        / "samples"
        / "mmlu_pro"
        / f"{seed}_{n}"
        / "samples.jsonl"
    )
    if sample_path.is_file():
        checks.append(Check("quality_samples", "pass", f"found {sample_path.relative_to(root)}"))
    else:
        checks.append(Check("quality_samples", "fail", f"missing {sample_path.relative_to(root)}"))

    registry = find_first(quality_registry_candidates(root, backend, safe_model))
    if registry is None:
        checks.append(Check("quality_registry", "fail", f"missing {backend} quality baseline registry"))
        return checks

    data = maybe_load_json(registry) or {}
    entries = ((data.get("datasets") or {}).get("mmlu_pro") or [])
    if isinstance(entries, dict):
        entries = [entries]
    exact = [
        e
        for e in entries
        if isinstance(e, dict) and int(e.get("seed", -1)) == seed and int(e.get("n", -1)) >= n
    ]
    if exact:
        acc = exact[0].get("accuracy")
        checks.append(
            Check(
                "quality_registry",
                "pass",
                f"found {registry.relative_to(root)} with mmlu_pro seed={seed} n>={n} accuracy={acc}",
            )
        )
    else:
        checks.append(
            Check(
                "quality_registry",
                "fail",
                f"{registry.relative_to(root)} has no mmlu_pro entry for seed={seed} n>={n}",
            )
        )
    return checks


def setup_commands(args: argparse.Namespace, safe_model: str, scenarios: list[str]) -> list[str]:
    scenario_arg = ",".join(scenarios)
    backend_safe = model_safe(args.quality_baseline_backend)
    q_registry_name = f"{safe_model}.json" if backend_safe == "vllm" else f"{safe_model}_{backend_safe}.json"
    return [
        "senpai/prepare_scoring_assets.sh "
        f"--scenario {scenario_arg} "
        f"--expected-gpu {args.expected_gpu!r} "
        f"--base-model {args.base_model!r} "
        f"--backend {args.speed_baseline_backend} "
        f"--quality-backend {args.quality_baseline_backend} "
        f"--dataset-seed {args.dataset_seed} "
        f"--quality-seed {args.quality_seed} "
        f"--mmlupro-n {args.mmlupro_n}",
        "python -m src.eval.inference.cache_samples --seed "
        f"{args.dataset_seed} --longbench-n 503 --mmlupro-n {args.mmlupro_n}",
        "INFERENCE_BENCH_ALLOW_HF_DOWNLOAD=1 "
        f"python senpai/materialize_requests.py --scenario {scenario_arg} --backend {args.speed_baseline_backend} "
        f"--base-model {args.base_model!r} --seed {args.dataset_seed}",
        "# Start the PyTorch baseline server on the same GPU/hardware, then run:",
        "python -m src.eval.inference.precompute_baseline "
        "--scenario-id <scenario_folder> "
        f"--base-model {args.base_model!r} --server-url http://127.0.0.1:9000 "
        f"--requests-file src/eval/inference/baselines/speed/{args.speed_baseline_backend}/<scenario_folder>/{safe_model}/requests.jsonl "
        f"--out-root src/eval/inference/baselines/speed/{args.speed_baseline_backend} "
        f"--registry src/eval/inference/baselines/speed/{args.speed_baseline_backend}/{safe_model}.json",
        "python -m src.eval.inference.precompute_quality_baseline "
        f"--backend {args.quality_baseline_backend} --base-model {args.base_model!r} "
        "--server-url http://127.0.0.1:9000 "
        f"--registry src/eval/inference/baselines/quality/{q_registry_name} "
        f"--seed {args.quality_seed} --mmlupro-n {args.mmlupro_n}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="all", help="A,B,C,D or all")
    parser.add_argument("--base-model", default=os.environ.get("INFERENCE_BENCH_BASE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--dataset-seed", type=int, default=int(os.environ.get("INFERENCE_BENCH_DATASET_SEED", "248")))
    parser.add_argument(
        "--quality-seed",
        type=int,
        default=int(os.environ.get("INFERENCE_BENCH_QUALITY_SEED", os.environ.get("INFERENCE_BENCH_DATASET_SEED", "248"))),
    )
    parser.add_argument("--mmlupro-n", type=int, default=int(os.environ.get("INFERENCE_BENCH_QUALITY_MMLUPRO_N", "500")))
    parser.add_argument("--speed-baseline-backend", default="torch")
    parser.add_argument("--quality-baseline-backend", default="torch")
    parser.add_argument("--expected-gpu", default="H100")
    parser.add_argument("--leaderboard-mode", action="store_true", help="Require H100-class hardware and full scoring assets.")
    parser.add_argument("--require-wandb", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--skip-runtime-imports", action="store_true")
    parser.add_argument("--allow-missing-vllm", action="store_true")
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--json-output", help="Optional JSON report path")
    args = parser.parse_args()

    root = repo_root()
    scenarios = selected_scenarios(args.scenario)
    safe_model = model_safe(args.base_model)

    checks: list[Check] = []
    if not args.skip_gpu:
        checks.append(check_hardware(args.expected_gpu, require_expected=args.leaderboard_mode))
    checks.append(check_wandb(require_wandb=args.require_wandb or args.leaderboard_mode))
    if not args.skip_runtime_imports:
        checks.append(check_runtime_imports(require_vllm=not args.allow_missing_vllm))
    speed_checks, requests_ready = check_speed_assets(root, scenarios, safe_model, args.speed_baseline_backend)
    checks.extend(speed_checks)
    checks.extend(check_quality_assets(root, safe_model, args.quality_baseline_backend, args.quality_seed, args.mmlupro_n))
    if not args.skip_tokenizer:
        checks.append(check_tokenizer(args.base_model, request_files_ready=requests_ready))

    report = {
        "status": "fail" if any(c.status == "fail" for c in checks) else "pass",
        "repo_root": str(root),
        "base_model": args.base_model,
        "model_safe": safe_model,
        "scenarios": scenarios,
        "checks": [c.as_dict() for c in checks],
        "setup_commands": setup_commands(args, safe_model, scenarios),
    }

    if args.json_output:
        Path(args.json_output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    for check in checks:
        print(f"[{check.status.upper()}] {check.name}: {check.detail}")
    if report["status"] != "pass":
        print("\nSuggested setup commands:")
        for command in report["setup_commands"]:
            print(f"  {command}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
