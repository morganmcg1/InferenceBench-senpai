#!/usr/bin/env python3
"""Workaround for a runner.py token-counting regression on transformers >=4.45.

Background
----------
`src/eval/inference/runner.py::_count_chat_tokens` calls
`tokenizer.apply_chat_template(..., tokenize=True)` and treats the result as a
list of token ids. On this transformers build that call returns a
`BatchEncoding`, whose `len()` is the number of keys (2: ``input_ids``,
``attention_mask``). All samples then look like they have 2 tokens, the
`min_input_tokens` filter rejects every cached LongBench-v2 sample, and
`_prepare_requests` raises ``No LongBench-v2 samples can satisfy the input
length range`` — so no scenario can run end-to-end on this pod.

This script bypasses that path by precomputing a deterministic per-scenario
``requests.jsonl`` under
``src/eval/inference/baselines/speed/{backend}/{scenario}/{model_safe}/``.
Runner falls back to that file when ``INFERENCE_BENCH_REQUESTS_FILE`` is unset,
and ``_load_requests_jsonl`` consumes it without going through the broken
`_count_chat_tokens` filter. We also set ``ignore_eos`` so generation goes the
full ``max_new_tokens`` (matches the synthetic-style speed contract).

The truncation logic inside ``_load_requests_jsonl`` *also* calls the broken
helper, but only to estimate ``base_tokens`` for the non-user portion. We set
``INFERENCE_BENCH_DATASET_HARD_TRUNCATE=0`` when invoking the evaluator so
truncation is skipped entirely and our pre-sized prompts are used as-is.

The file is intentionally limited to ``senpai/`` and the
``baselines/speed/...`` data dir; we do not modify any benchmark code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DIR = REPO_ROOT / "src" / "eval" / "inference"
TASKS_DIR = REPO_ROOT / "src" / "eval" / "tasks"

SCENARIO_TO_TASK = {
    "A": "inference_scenario_a_input_heavy",
    "B": "inference_scenario_b_output_heavy",
    "C": "inference_scenario_c_high_load",
    "D": "inference_scenario_d_general",
}


def model_safe(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)


def count_chat_tokens(tokenizer, messages: List[Dict[str, str]]) -> int:
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )
    return len(ids)


def truncate_user_to(
    tokenizer,
    messages: List[Dict[str, str]],
    target_chat_tokens: int,
) -> Tuple[List[Dict[str, str]], int]:
    """Head-truncate the last user message so the chat-template token count
    lands at exactly ``target_chat_tokens`` (or as close as the tokenizer
    boundary allows).
    """
    messages = [dict(m) for m in messages]
    user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            user_idx = i
            break
    if user_idx is None:
        raise ValueError("no user message in sample")

    original = messages[user_idx].get("content", "")
    cleared = [dict(m) for m in messages]
    cleared[user_idx]["content"] = ""
    base_tokens = count_chat_tokens(tokenizer, cleared)
    allowed = max(1, target_chat_tokens - base_tokens)

    user_ids = tokenizer.encode(original, add_special_tokens=False)
    if len(user_ids) <= allowed:
        return messages, count_chat_tokens(tokenizer, messages)

    # Binary search a token count that lands the *chat-template* total at or
    # just below target_chat_tokens (the chat template adds a few wrapper
    # tokens, so allowed is a lower bound, not exact).
    lo, hi = 1, len(user_ids)
    best_text = ""
    best_tokens = 0
    for _ in range(20):
        mid = (lo + hi) // 2
        head = user_ids[:mid]
        text = tokenizer.decode(head, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        cand = [dict(m) for m in messages]
        cand[user_idx]["content"] = text
        ct = count_chat_tokens(tokenizer, cand)
        if ct <= target_chat_tokens:
            best_text = text
            best_tokens = ct
            lo = mid + 1
        else:
            hi = mid - 1
        if lo > hi:
            break
    messages[user_idx]["content"] = best_text
    return messages, best_tokens


def load_scenario_config(scenario: str) -> Dict[str, Any]:
    task_dir = TASKS_DIR / SCENARIO_TO_TASK[scenario]
    return json.loads((task_dir / "scenario.json").read_text(encoding="utf-8"))


def find_samples_file(seed: int, pool_n: int) -> Path:
    base = INFERENCE_DIR / "baselines" / "samples" / "longbench_v2"
    if not base.is_dir():
        raise FileNotFoundError(base)
    cands: List[Tuple[int, Path]] = []
    for cand in base.glob(f"{seed}_*/samples.jsonl"):
        parent = cand.parent.name
        try:
            n = int(parent.split("_", 1)[1])
        except Exception:
            continue
        if n >= pool_n:
            cands.append((n, cand))
    if not cands:
        # fall back to the largest available pool
        for cand in base.glob(f"{seed}_*/samples.jsonl"):
            try:
                n = int(cand.parent.name.split("_", 1)[1])
                cands.append((n, cand))
            except Exception:
                continue
    if not cands:
        raise FileNotFoundError(f"no longbench_v2 cache for seed={seed}")
    cands.sort()
    return cands[0][1]


def build_for_scenario(tokenizer, scenario: str, model_id: str, out_root: Path) -> Path:
    cfg = load_scenario_config(scenario)
    synth = cfg.get("synthetic") or {}
    input_len = int(synth.get("input_len", 1024))
    output_len = int(synth.get("output_len", cfg.get("max_new_tokens", 256)))
    range_ratio = float(synth.get("range_ratio", 0.8))
    ignore_eos = bool(synth.get("ignore_eos", True))
    temperature = float(cfg.get("temperature", 0.2))
    seed = int(os.environ.get("INFERENCE_BENCH_DATASET_SEED", "248"))
    num_reqs = int(cfg.get("num_requests", 32))
    pool_n = max(num_reqs, int(os.environ.get("INFERENCE_BENCH_LONGBENCH_POOL_N", "503")))

    samples_path = find_samples_file(seed, pool_n)
    print(f"[build:{scenario}] using samples from {samples_path}")

    # Load all samples
    raw_samples: List[Dict[str, Any]] = []
    with samples_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw_samples.append(json.loads(line))

    min_input_tokens = max(1, int(math.ceil(input_len * range_ratio)))
    min_output_tokens = max(1, int(math.ceil(output_len * range_ratio)))
    min_output_tokens = min(min_output_tokens, output_len)

    selection_rng = random.Random(f"{seed}:longbench:{input_len}:{output_len}:{range_ratio}")
    input_rng = random.Random(f"{seed}:input:{input_len}:{output_len}:{range_ratio}")
    output_rng = random.Random(f"{seed}:output:{input_len}:{output_len}:{range_ratio}")

    # Pre-tokenize lengths so we can pick samples that have enough source content
    # to satisfy a target_input_tokens draw, then truncate.
    source_tokens: List[int] = []
    for r in raw_samples:
        msgs = r.get("messages") or []
        if not msgs:
            source_tokens.append(0)
            continue
        try:
            ct = count_chat_tokens(tokenizer, msgs)
        except Exception:
            ct = 0
        source_tokens.append(ct)
    eligible_idx = [i for i, t in enumerate(source_tokens) if t >= min_input_tokens]
    if not eligible_idx:
        raise RuntimeError(
            f"no samples for scenario {scenario}: min_input_tokens={min_input_tokens} "
            f"max_source={max(source_tokens) if source_tokens else 0}"
        )

    requests: List[Dict[str, Any]] = []
    for req_idx in range(num_reqs):
        target_input_tokens = input_rng.randint(min_input_tokens, input_len)
        # samples with source >= target
        cand = [i for i in eligible_idx if source_tokens[i] >= target_input_tokens]
        if not cand:
            # accept the largest available source
            cand = [max(eligible_idx, key=lambda j: source_tokens[j])]
        i = cand[selection_rng.randrange(len(cand))]
        record = raw_samples[i]
        msgs, realized_in = truncate_user_to(tokenizer, record.get("messages", []), target_input_tokens)
        target_output_tokens = output_rng.randint(min_output_tokens, output_len)
        req = {
            "sample_id": str(record.get("sample_id", f"{scenario}_{req_idx}")),
            "messages": msgs,
            "max_new_tokens": int(target_output_tokens),
            "temperature": temperature,
            "ignore_eos": ignore_eos,
            "target_input_token_count": int(target_input_tokens),
            "input_token_count": int(realized_in),
            "source_prompt_token_count": int(source_tokens[i]),
            "sampling_range_ratio": float(range_ratio),
        }
        requests.append(req)
        if req_idx % 32 == 0:
            print(f"[build:{scenario}] {req_idx}/{num_reqs} target_in={target_input_tokens} realized_in={realized_in} target_out={target_output_tokens}")

    out_dir = out_root / "vllm" / scenario / model_safe(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "requests.jsonl"
    with out_path.open("w") as fh:
        for r in requests:
            fh.write(json.dumps(r) + "\n")
    print(f"[build:{scenario}] wrote {len(requests)} requests to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="A,B,C", help="Comma-separated scenario ids")
    parser.add_argument("--model", default=os.environ.get("INFERENCE_BENCH_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"))
    parser.add_argument("--out-root", default=str(INFERENCE_DIR / "baselines" / "speed"))
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    out_root = Path(args.out_root)
    scenarios = [s.strip().upper() for s in args.scenarios.split(",") if s.strip()]
    for sc in scenarios:
        if sc not in SCENARIO_TO_TASK:
            print(f"[build] skip unknown scenario {sc}")
            continue
        build_for_scenario(tokenizer, sc, args.model, out_root)


if __name__ == "__main__":
    main()
