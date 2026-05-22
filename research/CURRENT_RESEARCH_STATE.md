# SENPAI Research State — InferenceBench (ib-20260522-r2)

- Current date and time: 2026-05-22 23:30 UTC (~57 min of 120 min budget used)
- Most recent research direction from human researcher team: none received yet
- Advisor branch: `ib-20260522-r2-advisor`
- W&B project: `wandb-applied-ai-team/inferencebench-senpai`
- Active students: r2-frieren, r2-fern, r2-tanjiro (3 logical students, 1 shared GPU)

## Hardware reality check (2026-05-22 23:05Z)

r2-tanjiro reported the GPU is **NVIDIA RTX PRO 6000 Black, 95 GiB VRAM (sm_89)**,
not the H100 80 GiB sm_90 quoted in `target/program.md`. Memory is more generous
(95 vs 80 GiB), but the reference snapshot's absolute speedups may not transfer
exactly. We continue to use the snapshot's `vLLM default, no agent` speedups
(A 1.25x, B 2.25x, C 48.69x, D 1.96x) as the only available calibration to
PyTorch.

## Evaluator bug — `_count_chat_tokens` (2026-05-22 23:12Z)

r2-fern diagnosed a real bug in `src/eval/inference/runner.py::_count_chat_tokens`:
under transformers 5.9.0, `tokenizer.apply_chat_template(..., tokenize=True)`
returns a `BatchEncoding` (len = 2) instead of a list of token ids, so every
LongBench-v2 sample is rejected by the eligibility filter and
`python evaluate.py` crashes with
`RuntimeError: No LongBench-v2 samples can satisfy the input length range`.

The fix (`return_dict=False`) is correct but `runner.py` is protected. We are
**not** patching `runner.py` in this launch. All three students use the
officially-supported escape hatch:

- Generate `requests.jsonl` externally per scenario (correct chat-template
  tokenization, same LongBench-v2 seed/pool/length filter), commit a generator
  helper under `senpai/research/prepare_scenario_*_requests.py` if useful.
- Pass to evaluator via `--requests-file` or `INFERENCE_BENCH_REQUESTS_FILE`.

This also makes `precompute_baseline.py` unusable since it routes through the
same broken `_prepare_requests`.

## Missing PyTorch baseline metrics file

There is no `pytorch_baseline_metrics.json` in `target/`. Each student must
build a derived PyTorch baseline:

1. Run vLLM-default starting-point launcher (`src/starting_points/vllm_running/start_server.sh`)
   first under the same `--requests-file` → `metrics_default.json`.
2. Run the experimental launcher → `metrics_exp.json`.
3. Derived PyTorch primary = (vLLM-default raw primary) ÷ (reference vLLM-default
   speedup for that scenario). Pass as `--baseline-primary` to `summarize_metrics.py`.
4. SENPAI-RESULT reports both raw values, derived PyTorch baseline, and
   `scenario/X/speedup_over_pytorch`.

## Current PR state (2026-05-22 23:30Z)

| PR | Student | Scenario | State | Notes |
|---|---|---|---|---|
| #7 | r2-frieren | B (long-decode TPOT) | WIP, no commits | nudged for status + told about bug |
| #8 | r2-fern | A (long-prefill TTFT) | WIP, launcher pushed 23:27 | following workaround (a) |
| #9 | r2-tanjiro | C (high-load throughput) | WIP, 2 launchers pushed 23:02 | waiting for GPU slot |

## Time bookkeeping

- 23:30Z, ~57 min used, ~63 min remaining.
- Each student now has a 2-leg measurement plan (vLLM-default + experiment) plus
  the external `requests.jsonl` generator step. That is realistically 25–35 min
  per student of GPU+code time. The shared GPU is the binding constraint.

## Risks and mitigations

- **r2-frieren stale.** No commits or comments after 55 min. Just sent a status
  ping with the eval-bug heads-up. Watch the next cycle; if still silent at +20
  min, escalate or reassign.
- **GPU contention.** Three students × 2 launches each = 6 benchmark cycles on
  a single GPU in ~60 min. Some students may not finish a full eval. If only one
  scenario lands a clean result before the window closes, that is the round-1
  winner.
- **Quality gate under FP8 weight quant.** r2-fern's arm 3 (drop weight FP8) is
  the safety valve if quality drops below 0.95.
- **Hardware deviation from reference snapshot.** Numbers below the snapshot's
  reference speedup do not necessarily mean the hypothesis failed — could just
  be H100→RTX PRO delta. Compare to vLLM-default on the same hardware.

## Potential next research directions (once any scenario completes)

- Engine A/B: SGLang vs vLLM under the same scenario constraints once one
  vLLM-only winner is locked.
- Speculative decoding sweep on scenario D once B converges.
- Attention backend ablation: FLASHINFER vs FLASH_ATTN vs TRITON_ATTN at the
  best per-scenario operating point.
- Cross-scenario confirmation (A-D geomean) once any scenario beats its
  reference number.
