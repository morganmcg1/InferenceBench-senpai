# SENPAI Research State — ib-20260524-ready-r5

- **Last updated:** 2026-05-24 (advisor boot)
- **Most recent human directive:** (none on this branch; using `program.md`
  research contract and the operator-set 2-hour shakedown clock)
- **Active hardware:** RTX PRO 6000 Blackwell (96 GB), shakedown only
- **Active model:** Mistral-7B-Instruct-v0.3

## Current research focus

First-wave launcher attack across three scenarios in parallel, one student per
scenario, on a shared 1-GPU pod. Goal: replace the vLLM default starting point
with a tuned vLLM launcher for each scenario and establish per-scenario
RTX PRO 6000 best-known speedups in this ledger.

Three coordinated arms:

1. **Scenario B — output-heavy decode (TPOT).** vLLM + FP8 weights +
   FP8 KV cache + n-gram speculative decoding + CUDA graphs. Highest
   theoretical upside on the public H100 board (15.23× via SMAC3 search).
2. **Scenario A — input-heavy prefill (TTFT).** vLLM + chunked prefill +
   large `max-num-batched-tokens` + FP8 weights. Aim at prefill-throughput.
3. **Scenario D — general/balanced.** vLLM + FP8 weights + KV-cache FP8 +
   moderate batch + speculative tokens. Balanced settings across the burst@4
   profile with both TTFT and TPOT contributions.

Scenario C is deferred for round 2: the public board already shows SGLang
default at 51.12× req/s, and beating it is a more delicate optimization that
benefits from learning from the round-1 results first.

## Shared-GPU coordination contract

- One **heavy full eval at a time** on the shared pod. Students post
  `SLOT-FREE` in their PR comments when their server is fully torn down.
- Each student may do up to 2 quick `evaluate.py --quick` arms before their
  full eval, but never while another student holds the slot.
- Students may run smoke tests, launcher prep, log analysis, materializing
  request files, and writing research notes off-GPU at any time.

## Potential next research directions

- Speculative-decoding ablation: n-gram window size, draft tokens 3/5/7, with
  and without FP8 KV cache for Scenario B.
- FlashInfer attention backend vs FLASH_ATTN on Blackwell for Scenario A; this
  has historically been the dominant lever for input-heavy TTFT.
- Scenario C attack: SGLang with radix prefix caching tuned for the high-load
  256-request profiles, vs vLLM with very large `max-num-seqs` and FP8 KV.
- AWQ / GPTQ quantized Mistral-7B-Instruct-v0.3 variants — only if the quality
  gate stays ≥ 0.95× baseline.
- TensorRT-LLM launcher for Scenarios A/B as a stretch direction once the vLLM
  family is well-characterized.
- Cross-scenario A–D confirmation runs for the round-1 winners to support a
  later aggregate geomean speedup result.
