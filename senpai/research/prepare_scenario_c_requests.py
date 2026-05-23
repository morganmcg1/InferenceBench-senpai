#!/usr/bin/env python3
"""Build a precomputed requests.jsonl for scenario C, bypassing the
`_count_chat_tokens` bug in runner.py.

Mirrors the seeded RNG sequence in `_prepare_requests` so the request
distribution matches what the runner intends. Uses
`apply_chat_template(tokenize=True)["input_ids"]` (not `len(...)`) for
accurate token counts.
"""
import json
import os
import random
import hashlib
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/hf_cache/hub"))

from transformers import AutoTokenizer

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
SEED = 248
INPUT_LEN = 1024
OUTPUT_LEN = 1024
RANGE_RATIO = 0.8
NUM_REQS = 256
TEMP = 0.3
IGNORE_EOS = True
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]
SAMPLES_PATH = Path(os.environ.get(
    "INFERENCE_BENCH_LONGBENCH_SAMPLES",
    str(_REPO_ROOT / "src/eval/inference/baselines/samples/longbench_v2/248_503/samples.jsonl"),
))

OUTPUT = Path(os.environ.get(
    "INFERENCE_BENCH_REQUESTS_OUTPUT",
    "/tmp/scen_c/inference/baselines/speed/torch/inference_scenario_c_high_load/requests.jsonl",
))


def count_tokens(messages, tok):
    out = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    if hasattr(out, "input_ids"):
        ids = out["input_ids"] if not isinstance(out["input_ids"], list) else out["input_ids"]
    elif isinstance(out, dict):
        ids = out["input_ids"]
    else:
        ids = out
    return len(ids)


def truncate_user_head(messages, tok, target_input_tokens):
    """Head-truncate the (last) user message so the chat-rendered prompt
    has approximately target_input_tokens. Matches runner._truncate_messages
    semantics but uses a working tokenizer.
    """
    # Find last user message.
    user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            user_idx = i
            break
    if user_idx is None:
        return messages, count_tokens(messages, tok)

    original = messages[user_idx].get("content", "")
    # Determine the per-message overhead by replacing with empty content.
    blanked = [dict(m) for m in messages]
    blanked[user_idx]["content"] = ""
    base = count_tokens(blanked, tok)
    allowed = max(0, target_input_tokens - base)

    # Tokenize user content alone and head-truncate.
    user_tokens = tok.encode(original, add_special_tokens=False)
    if len(user_tokens) <= allowed:
        # Fits; return as-is.
        return messages, count_tokens(messages, tok)

    truncated_tokens = user_tokens[:allowed]
    truncated_text = tok.decode(truncated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    out_msgs = [dict(m) for m in messages]
    out_msgs[user_idx]["content"] = truncated_text
    return out_msgs, count_tokens(out_msgs, tok)


def content_hash(req):
    payload = {
        "messages": req.get("messages"),
        "max_new_tokens": req.get("max_new_tokens"),
        "temperature": req.get("temperature"),
        "require_json": req.get("require_json", False),
        "ignore_eos": req.get("ignore_eos", False),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main():
    print(f"[builder] loading tokenizer {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

    min_input = max(0, int(INPUT_LEN * RANGE_RATIO + 0.999))  # ceil
    min_output = max(1, int(OUTPUT_LEN * RANGE_RATIO + 0.999))
    print(f"[builder] min_input={min_input} max_input={INPUT_LEN} min_output={min_output} max_output={OUTPUT_LEN}")

    # Load + token-count pool.
    print(f"[builder] indexing {SAMPLES_PATH}")
    eligible = []  # (record, messages, source_prompt_tokens)
    with SAMPLES_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            msgs = [dict(m) for m in rec.get("messages", [])]
            n = count_tokens(msgs, tok)
            if n < min_input:
                continue
            eligible.append((rec, msgs, n))
    print(f"[builder] eligible pool size: {len(eligible)} / {503}")
    if not eligible:
        raise SystemExit("ERROR: no eligible samples")

    # Mirror runner RNGs.
    sel_rng = random.Random(f"{SEED}:longbench:{INPUT_LEN}:{OUTPUT_LEN}:{RANGE_RATIO}")
    in_rng = random.Random(f"{SEED}:input:{INPUT_LEN}:{OUTPUT_LEN}:{RANGE_RATIO}")
    out_rng = random.Random(f"{SEED}:output:{INPUT_LEN}:{OUTPUT_LEN}:{RANGE_RATIO}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUTPUT.open("w") as out_fh:
        for req_idx in range(NUM_REQS):
            target_input = in_rng.randint(min_input, INPUT_LEN)
            cands = [(r, m, s) for (r, m, s) in eligible if s >= target_input]
            if not cands:
                raise SystemExit(f"no candidate for req_idx={req_idx} target_input={target_input}")
            rec, msgs, src_n = cands[sel_rng.randrange(len(cands))]
            truncated, realized = truncate_user_head(msgs, tok, target_input)
            # Guard: pull in a few tokens if overshoot, else accept.
            if realized > target_input:
                # Try a slightly smaller target to land within range.
                for slack in (1, 2, 4, 8, 16):
                    truncated2, realized2 = truncate_user_head(msgs, tok, target_input - slack)
                    if realized2 <= target_input:
                        truncated, realized = truncated2, realized2
                        break
            if not (min_input <= realized <= target_input):
                # Last resort: clamp into range. Doesn't have to be exact for speed eval purposes.
                print(f"[builder] WARN req_idx={req_idx} target={target_input} realized={realized}")
                if realized < min_input:
                    # Re-pick a longer source.
                    long_cands = [(r, m, s) for (r, m, s) in eligible if s >= target_input * 2]
                    if long_cands:
                        rec, msgs, src_n = long_cands[sel_rng.randrange(len(long_cands))]
                        truncated, realized = truncate_user_head(msgs, tok, target_input)
            max_new = out_rng.randint(min_output, OUTPUT_LEN)
            req = {
                "messages": truncated,
                "max_new_tokens": max_new,
                "temperature": TEMP,
                "require_json": False,
                "gold_answer": None,
                "sample_id": str(rec.get("sample_id", "")),
                "parse_mode": "",
                "ignore_eos": IGNORE_EOS,
                "target_input_token_count": target_input,
                "input_token_count": realized,
                "source_prompt_token_count": src_n,
                "sampling_range_ratio": RANGE_RATIO,
            }
            req["content_hash"] = content_hash(req)
            out_fh.write(json.dumps(req, ensure_ascii=False) + "\n")
            written += 1
            if (req_idx + 1) % 32 == 0:
                print(f"[builder] wrote {req_idx + 1}/{NUM_REQS}")
    print(f"[builder] DONE wrote {written} requests to {OUTPUT}")


if __name__ == "__main__":
    main()
