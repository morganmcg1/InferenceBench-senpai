# SENPAI Research State — `ib-20260528-scen-c-r1`

- Date: 2026-05-28 (updated ~T+95, launch entering review/scorekeeping window)
- Active research tag: `ib-20260528-scen-c-r1`
- Advisor branch: `ib-20260528-scen-c-r1`
- Scope: Scenario C only (high-load, geomean throughput across burst/poisson/constant).
- Students: scen-c-frieren, scen-c-fern (1 GPU shared across both via `senpai/gpu_slot.py`).
- Hardware: 1× RTX PRO 6000 (shakedown mode).

## Most recent human research direction

- Operator launch: paper-parity Scenario C trial, ~2h budget, two logical students on one benchmark
  GPU. Require terminal results to come from supervised relaunch + `validate_result.py`.

## Live baseline

**23.98x** speedup_over_pytorch (Scenario C geomean). **PR #165 merged**. vLLM 0.11.0 + n-gram speculative decoding (num_speculative_tokens=5, prompt_lookup_min=3, prompt_lookup_max=5) on top of PR #161 Arm A config (max_num_seqs=384, max_num_batched_tokens=16384, gpu_mem_util=0.92, chunked_prefill ON). W&B: mj8f07f0. Quality ratio 1.000 (n=500, exact match). Self-contained launcher.

Lineage:
- Rank 1: PR #165 vLLM+ngram 23.98x (current)
- Rank 2: PR #163 SGLang 22.18x
- Rank 3: PR #161 vLLM Arm A 20.84x

## Research findings worth carrying forward

- **vLLM 0.11.0 V1: `--no-enable-chunked-prefill` is a no-op** (engine forces chunked_prefill=True in arg_utils.py:1548 for non-pooling models). Future scenarios A/B/D should not test this flag.
- **n-gram speculative decoding wins on long-decode regimes** with structured/instruction-tuned models — Scenario C's output_len=1024 + ignore_eos=true is the right environment. Quality preserved exactly (ratio 1.000).
- **SGLang is competitive with vLLM on RTX PRO 6000** when configured with triton attention + pytorch sampler. The FlashInfer/SM120 incompatibility doesn't prevent strong baselines.
- **SGLang launcher in PR #163 uses env-var-controlled defaults**; reproduce requires setting SGLANG_* env vars. Open follow-up: hardcode defaults so the launcher is self-contained.

## Current state

- ~T+95 of 120 min launch budget. Entering review/scorekeeping window (~25 min remaining, ~10-15 reserved).
- All Scenario C PRs in this launch are now resolved:
  - PR #161 merged (rank 3, 20.84x)
  - PR #163 merged (rank 2, 22.18x)
  - PR #165 merged (**rank 1, 23.98x** — current winner)
  - PR #167 closed (no commits since assignment; insufficient wall time for full eval; would have needed to clear 23.98x not 22.18x)
- frieren is idle after the PR #165 merge. fern's PR #167 was stuck — no commits, no comments since 17:39, and `--min-remaining-s 1800` would refuse a fresh full eval at this point.
- No new assignments planned for the remaining budget; the launcher-defaults hardcoding follow-up (see below) is non-urgent and not worth a partial-attempt PR in the final ~10 min.

## Active PRs

_None._ Launch is in scorekeeping phase.

## Potential next research directions (for future launches)

- Higher num_speculative_tokens (7, 9) — diminishing returns expected.
- vLLM with `method=eagle` if a Mistral-7B EAGLE draft checkpoint becomes available.
- SGLang + speculative decoding (EAGLE/MEDUSA) to combine the two leads.
- FP8 weight quantization (BF16 KV preserved) — could free VRAM for higher concurrency.
- Try the SM120-compatible flashinfer build that recent SGLang patches reportedly support.

## Stop rules / hard limits

- Reserve last 10-15 minutes for review + `BASELINE.md` update.
- No full evaluation may start unless `gpu_slot.py --mode full --min-remaining-s 1800` passes.
- Quality gate failures are not winners.
- All terminal updates must come through `validate_result.py`.
- Do NOT score Scenarios A, B, D in this launch.
