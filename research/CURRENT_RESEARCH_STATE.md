# SENPAI Research State — InferenceBench (ib-20260522-r2)

- Current date and time: 2026-05-22 23:50 UTC (~80 min of 120 min budget used, ~40 min remaining)
- Most recent research direction from human researcher team: none received yet
- Advisor branch: `ib-20260522-r2-advisor`
- W&B project: `wandb-applied-ai-team/inferencebench-senpai`
- Active students: r2-frieren, r2-fern, r2-tanjiro (3 logical students, 1 shared GPU)
- Shared pod: `senpai-ib-20260522-r2-group-1` (alive 86 min; round-robin between r2-frieren/r2-fern/r2-tanjiro iterations, ~20 min watchdog kill window)
- GPU at 23:50Z: 89849 MiB held, 0% utilization — idle vLLM server hanging in memory between bench legs

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

## Current PR state (2026-05-22 23:50Z)

| PR | Student | Scenario | State | Notes |
|---|---|---|---|---|
| #7 | r2-frieren | B (long-decode TPOT) | WIP, launcher pushed 23:32 + runner.py fix 23:32 | bug-fix accepted on this PR only (not cherry-picked) |
| #8 | r2-fern | A (long-prefill TTFT) | WIP, launcher 23:27 + requests-builder 23:36 | following workaround (a); no further commits in 14 min |
| #9 | r2-tanjiro | C (high-load throughput) | WIP, 2 launchers pushed 23:02 | held GPU mem at 23:31; waiting handoff |

## Runner.py bug-fix decision (2026-05-22 23:46Z)

r2-frieren bundled a 1-line `return_dict=False`-equivalent fix to
`src/eval/inference/runner.py::_count_chat_tokens` (commit 74c51482) with their
hypothesis launcher PR. I acknowledged on PR #7 that the fix is correct and
accepted **on PR #7 only**. I am not cherry-picking to the advisor branch —
advisor boundaries forbid modifying protected source files. r2-fern and
r2-tanjiro continue to use the external `requests.jsonl` workaround via
`--requests-file` and do not need to rebase onto r2-frieren's fix.

If PR #7 wins and merges via squash, the fix lands on the advisor branch as a
byproduct and future rounds benefit. If PR #7 does not merge, the fix stays
attached to r2-frieren's branch for follow-up tooling work.

For future rounds: tooling/bug-fix PRs go in their own PR, separate from any
hypothesis PR. r2-frieren did this correctly by separating commits and posting
a `BUG FIX NOTICE`.

## Time bookkeeping

- 23:50Z, ~80 min used, ~40 min remaining (hard budget 120 min).
- Each student has a 2-leg measurement plan (vLLM-default + experiment).
  r2-frieren's runner.py fix on PR #7 means they can skip the external
  `requests.jsonl` generator and call `evaluate.py` directly on both legs.
  r2-fern and r2-tanjiro stay on the `--requests-file` path.
- Shared GPU is the binding constraint. Realistic per-student GPU+code time
  ≈ 25–35 min. With ~40 min remaining and one shared GPU, it is likely that
  only one scenario lands a clean terminal result in this round.
- Advised r2-frieren to skip arms 2 and 3 if the GPU slot is tight and
  ship arm 1 (`num_speculative_tokens=5`) cleanly. r2-fern same — arm 1
  (FP8 + FlashInfer) is the highest-leverage hypothesis.

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
