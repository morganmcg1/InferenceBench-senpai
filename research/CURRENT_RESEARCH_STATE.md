# SENPAI Research State

- **Date/time:** 2026-05-25 ~08:30 UTC (in-session update)
- **Research tag:** ib-20260525-one3-r1
- **Hardware:** 1x RTX PRO 6000 Blackwell ~96GB VRAM (shakedown; leaderboard claims need H100)
- **Time budget:** ~55 min remaining (budget runs to ~09:29 UTC)

## Most recent research direction from human researcher team

No human-team directives received this session.

## Current research focus and themes

**Active PR:** #101 (frieren, Scenario B n-gram speculative decoding)

PR #100 (Scenario A) merged with **1.2433x speedup** (TTFT.p50 0.3527s vs PyTorch
0.4385s). Key finding: chunked vs monolithic prefill neutral at concurrency 1
-- no decode to interleave with. Baseline established for Scenario A.

Now pushing into Scenario B: the highest-headroom scenario (public HPO 15.23x
vs vLLM default 2.25x). N-gram speculative decoding is the primary lever:
concurrency 1 with 8K output is the textbook best case for batch-1 spec dec.
Single arm, no sweep -- time is critical.

### Active experiments

| PR | Scenario | Student | Hypothesis | Status |
|---|---|---|---|---|
| #101 | B (TPOT) | frieren | vLLM n-gram spec dec (5 tokens, FlashAttn) | WIP |

### Completed / merged

| PR | Scenario | Speedup | Key finding |
|---|---|---|---|
| #100 | A (TTFT) | **1.2433x** | Chunked vs monolithic neutral at conc 1; FlashAttn is main gain |

## Potential next research directions

### Within this session (time permitting after PR #101)

- **Scenario B follow-up:** if PR #101 n-gram acceptance rate is low (<0.2),
  try a fresh arm with num_speculative_tokens=3 or fall back to pure
  FlashAttn baseline for B.
- **Scenario A Triton kernel ablation:** replace FLASH_ATTN with TRITON_ATTN
  on the PR #100 launcher -- measures whether Blackwell GPU architecture prefers
  the Triton kernel for prefill.

### Post-session or H100 follow-ups

- **FP8 weight quantization:** need quality gate validation. Could give 1.5-2x
  TPOT gain if MMLU-Pro passes.
- **Scenario D (balanced, conc 4):** port the A+B winner config. Chunked prefill
  will matter here (concurrent decode exists at conc 4). Likely 3-5x achievable.
- **SGLang for Scenario C:** vLLM default already dominates C; SGLang RadixAttention
  is a long-shot worth one quick probe on H100.
- **H100 confirmation:** repeat PR #100 and PR #101 winners on H100 to convert
  shakedown numbers into leaderboard-comparable results.

## Key invariants to preserve

- All results must use pre-staged scoring assets:
  senpai/require_scoring_preflight.sh --import-dir /mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248
- Every terminal result needs full evaluate.py run, W&B log, and clean relaunch.
- RTX PRO 6000 shakedown: no FlashInfer unless explicitly tested; no FP8 KV
  cache unless proven on this hardware.
- BASELINE.md is advisor-owned and updated only from verified terminal runs.
- VRAM ceiling: PR #100 VRAM = 90.2GiB at 0.90 util. Any H100 80GB target
  needs --gpu-memory-utilization <= 0.75.
