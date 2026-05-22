#!/usr/bin/env python3
"""Build a precomputed requests.jsonl for scenario A, mimicking _prepare_requests.

Workaround for runner.py _count_chat_tokens bug under transformers 5.9.0 where
apply_chat_template(tokenize=True) returns BatchEncoding (len=2) instead of a
token list. We produce the same {messages, max_new_tokens, ...} stream that
_prepare_requests would, using the fast tokenizer with return_dict=False.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from transformers import AutoTokenizer


def count_chat_tokens(messages, tokenizer):
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=False
    )
    return len(ids)


def truncate_user_message(messages, tokenizer, max_input_tokens, *, keep="head"):
    msgs = [dict(m) for m in messages]
    user_idx = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            user_idx = i
            break
    if user_idx is None:
        return msgs
    original = msgs[user_idx].get("content", "")
    no_user = [dict(m) for m in msgs]
    no_user[user_idx]["content"] = ""
    base = count_chat_tokens(no_user, tokenizer)
    allowed = max_input_tokens - base
    if allowed <= 0:
        msgs[user_idx]["content"] = ""
        return msgs
    user_ids = tokenizer.encode(original, add_special_tokens=False)
    if len(user_ids) <= allowed:
        return msgs
    if keep == "head":
        keep_ids = user_ids[:allowed]
    else:
        keep_ids = user_ids[-allowed:]
    msgs[user_idx]["content"] = tokenizer.decode(
        keep_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return msgs


def request_content_hash(req):
    payload = {
        "messages": req.get("messages"),
        "max_new_tokens": req.get("max_new_tokens"),
        "temperature": req.get("temperature"),
        "require_json": req.get("require_json", False),
        "ignore_eos": req.get("ignore_eos", False),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples-file", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-requests", type=int, default=128)
    p.add_argument("--input-len", type=int, default=8192)
    p.add_argument("--output-len", type=int, default=1024)
    p.add_argument("--range-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=248)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--ignore-eos", action="store_true", default=True)
    p.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.3")
    args = p.parse_args()

    target_input_len = args.input_len
    output_len = args.output_len
    range_ratio = args.range_ratio
    seed = args.seed

    min_input_tokens = max(0, int(math.ceil(target_input_len * range_ratio)))
    min_output_tokens = max(1, int(math.ceil(output_len * range_ratio)))
    min_output_tokens = min(min_output_tokens, output_len)

    cache = "/workspace/home-r2-fern/hf_cache/hub"
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, cache_dir=cache)
    print(f"tokenizer: {type(tok).__name__}")

    # Load eligible samples with correct chat-template token counts.
    eligible = []
    with open(args.samples_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = [dict(m) for m in record.get("messages", [])]
            n = count_chat_tokens(messages, tok)
            if n < min_input_tokens:
                continue
            eligible.append(({"record": record, "messages": messages}, n))

    print(f"eligible: {len(eligible)}")

    selection_rng = random.Random(f"{seed}:longbench:{target_input_len}:{output_len}:{range_ratio}")
    input_rng = random.Random(f"{seed}:input:{target_input_len}:{output_len}:{range_ratio}")
    output_rng = random.Random(f"{seed}:output:{target_input_len}:{output_len}:{range_ratio}")

    with open(args.out, "w") as out:
        for req_idx in range(args.num_requests):
            target_input_tokens = input_rng.randint(min_input_tokens, target_input_len)
            cands = [(it, src) for it, src in eligible if src >= target_input_tokens]
            if not cands:
                max_src = max(src for _, src in eligible)
                raise RuntimeError(
                    f"No candidate at req={req_idx} target={target_input_tokens} max_src={max_src}"
                )
            item, src = cands[selection_rng.randrange(len(cands))]
            record = item["record"]
            messages = [dict(m) for m in item["messages"]]
            messages = truncate_user_message(messages, tok, target_input_tokens, keep="head")
            realized = count_chat_tokens(messages, tok)
            assert min_input_tokens <= realized <= target_input_tokens, (
                f"realized={realized} not in [{min_input_tokens},{target_input_tokens}]"
            )
            max_new_tokens = output_rng.randint(min_output_tokens, output_len)

            req = {
                "messages": messages,
                "max_new_tokens": max_new_tokens,
                "temperature": args.temperature,
                "require_json": False,
                "gold_answer": None,
                "sample_id": str(record.get("sample_id", "")),
                "parse_mode": "",
                "ignore_eos": True,
                "target_input_token_count": target_input_tokens,
                "input_token_count": realized,
                "source_prompt_token_count": src,
                "sampling_range_ratio": range_ratio,
            }
            req["content_hash"] = request_content_hash(req)
            out.write(json.dumps(req, ensure_ascii=False) + "\n")

    print(f"wrote {args.num_requests} requests to {args.out}")


if __name__ == "__main__":
    main()
