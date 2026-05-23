#!/usr/bin/env python3
"""Materialize deterministic speed request files without editing the evaluator."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
SCENARIOS = {
    "A": "inference_scenario_a_input_heavy",
    "B": "inference_scenario_b_output_heavy",
    "C": "inference_scenario_c_high_load",
    "D": "inference_scenario_d_general",
}


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


def robust_count_chat_tokens(messages: list[dict[str, str]], tokenizer: Any) -> int:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            tokens = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
            return tokenized_length(tokens)
        except Exception:
            pass
    joined = "\n".join(f"{m.get('role')}: {m.get('content', '')}" for m in messages)
    return len(tokenizer.encode(joined, add_special_tokens=False))


def write_requests_jsonl(path: Path, requests_list: list[dict[str, Any]], runner: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for req in requests_list:
            handle.write(
                json.dumps(
                    {
                        "messages": req.get("messages"),
                        "gold_answer": req.get("gold_answer"),
                        "sample_id": req.get("sample_id"),
                        "max_new_tokens": req.get("max_new_tokens"),
                        "temperature": req.get("temperature"),
                        "require_json": req.get("require_json", False),
                        "parse_mode": req.get("parse_mode", ""),
                        "ignore_eos": req.get("ignore_eos", False),
                        "target_input_token_count": req.get("target_input_token_count"),
                        "input_token_count": req.get("input_token_count"),
                        "source_prompt_token_count": req.get("source_prompt_token_count"),
                        "sampling_range_ratio": req.get("sampling_range_ratio"),
                        "content_hash": req.get("content_hash") or runner._request_content_hash(req),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def output_path(root: Path, backend: str, scenario_folder: str, safe_model: str, out_root: str) -> Path:
    if out_root:
        return Path(out_root).expanduser() / backend / scenario_folder / safe_model / "requests.jsonl"
    return root / "src" / "eval" / "inference" / "baselines" / "speed" / backend / scenario_folder / safe_model / "requests.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="all", help="A,B,C,D or all")
    parser.add_argument("--backend", action="append", default=[], help="Output backend namespace; repeatable. Default: torch")
    parser.add_argument("--base-model", default=os.environ.get("INFERENCE_BENCH_BASE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("INFERENCE_BENCH_DATASET_SEED", "248")))
    parser.add_argument("--limit", type=int, default=None, help="Optional request limit for debugging only.")
    parser.add_argument("--out-root", default="", help="Optional root; defaults to src/eval/inference/baselines/speed")
    parser.add_argument("--allow-download", action="store_true", help="Allow tokenizer/dataset downloads if not cached.")
    args = parser.parse_args()

    if args.allow_download:
        os.environ["INFERENCE_BENCH_ALLOW_HF_DOWNLOAD"] = "1"

    root = repo_root()
    sys.path.insert(0, str(root))
    from src.eval.inference import runner  # pylint: disable=import-outside-toplevel

    # Keep the protected evaluator unchanged, but use a patched token counter
    # while materializing request files. This avoids the transformers
    # BatchEncoding len()==2 trap seen in the shakedown run.
    runner._count_chat_tokens = robust_count_chat_tokens  # type: ignore[attr-defined]  # pylint: disable=protected-access

    tokenizer = runner._get_tokenizer(args.base_model)  # pylint: disable=protected-access
    if tokenizer is None:
        raise SystemExit(
            "Tokenizer unavailable. Pre-cache the model tokenizer or rerun with "
            "INFERENCE_BENCH_ALLOW_HF_DOWNLOAD=1 / --allow-download."
        )
    max_model_len = runner._get_model_max_len(args.base_model)  # pylint: disable=protected-access

    backends = args.backend or ["torch"]
    safe_model = model_safe(args.base_model)
    for scenario in selected_scenarios(args.scenario):
        folder = SCENARIOS[scenario]
        task_dir = root / "src" / "eval" / "tasks" / folder
        config = runner.load_scenario_config(task_dir)
        config["dataset_seed"] = args.seed
        requests_list, _ = runner._prepare_requests(  # pylint: disable=protected-access
            config,
            limit=args.limit,
            tokenizer=tokenizer,
            max_model_len=max_model_len,
        )
        for backend in backends:
            path = output_path(root, backend, folder, safe_model, args.out_root)
            write_requests_jsonl(path, requests_list, runner)
            meta_path = path.with_suffix(".meta.json")
            meta_path.write_text(
                json.dumps(
                    {
                        "scenario": scenario,
                        "scenario_folder": folder,
                        "backend_namespace": backend,
                        "base_model": args.base_model,
                        "dataset_seed": args.seed,
                        "request_count": len(requests_list),
                        "materialized_by": "senpai/materialize_requests.py",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"[requests] {scenario} wrote {len(requests_list)} rows to {path}")


if __name__ == "__main__":
    main()
