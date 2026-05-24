# SENPAI Research State — ib-20260524-leasefix-r4

- **As of:** 2026-05-24 22:00 UTC (start of 2 h SENPAI window)
- **Most recent human-team directive:** none yet for this tag.
- **Hardware:** NVIDIA RTX PRO 6000 Blackwell shakedown run (not
  leaderboard-comparable). Single GPU shared across 3 student logical
  workers in one pod.
- **Target:** `morganmcg1/InferenceBench-senpai` — InferenceBench LLM serving
  benchmark, base model Mistral-7B-Instruct-v0.3, scenarios A/B/C/D, MMLU-Pro
  quality gate at tau=0.95.

## Current research focus

This is a fresh shakedown run: we have a PyTorch raw baseline imported from
`rtxpro6000-seed248` scoring assets and no measured vLLM (or other engine)
launcher results yet. Round 1 establishes a clean vLLM-on-Blackwell
baseline across three scenarios in parallel:

- **Scenario A — input-heavy 8k/1k prefill TTFT.** Tune for fast single-shot
  long-prefill. Initial lever set: FLASH_ATTN backend, large
  `--max-num-batched-tokens` (≥16384), no chunked prefill, no prefix cache,
  CUDA graphs on, `--gpu-memory-utilization 0.92`.
- **Scenario B — output-heavy 1k/8k decode TPOT.** Tune for fast
  single-stream decode. Initial lever set: FLASH_ATTN, small
  `--max-num-seqs`, CUDA graphs on, no chunked prefill or prefix cache.
- **Scenario D — balanced 4k/2k conc-4 burst.** Tune for mixed serving.
  Initial lever set: FLASH_ATTN, chunked prefill on, mid-range
  `--max-num-seqs 32` and `--max-num-batched-tokens 8192`.

Scenario C is held back from round 1: the public reference shows vLLM
default already at or above the SMAC3 search winner there (48.69x vs
46.70x on H100). The PyTorch baseline at concurrency is the weakest, so
any reasonable vLLM config wins — better to commit GPU minutes elsewhere
this round.

We are **not** turning on FP8 KV cache, FP8 quantization, FlashInfer, or
speculative decoding in round 1. Each of those is a strong individual
lever and deserves its own controlled PR with a clean comparison.

## Active PRs

- **#83 frieren / sc-A-vllm-fastprefill** — Scenario A opening recipe.
- **#84 fern / sc-B-vllm-decode** — Scenario B opening recipe.
- **#86 tanjiro / sc-D-vllm-balanced** — Scenario D opening recipe.

GPU slot order this round: frieren → tanjiro → fern (B has the longest
wall-clock, goes last).

## Potential next research directions

Round 2 candidates (post round-1 baseline measurement):

1. **Speculative decoding for B/D.** vLLM supports n-gram speculative
   decoding which directly targets TPOT for long generation. Quality
   gate must pass.
2. **FP8 KV cache for B/D.** Bandwidth-bound decode benefits from smaller
   KV cache. RTX PRO 6000 supports FP8; controlled hardware-path PR with
   strict quality-gate evidence.
3. **Prefix caching for A** if there is any prompt-prefix overlap in
   LongBench-v2 (likely small; cheap to test).
4. **Block-size sweep (16 vs 32)** once we have a clean round-1 baseline.
5. **TensorRT-LLM** or **SGLang** as alternative engines, scenario-by-
   scenario, only if vLLM hits a clear ceiling on one scenario.
6. **Scenario C tuning** if round 1 finishes early — focus on
   `request_throughput_req_per_s` across the three traffic profiles, with
   larger `--max-num-seqs` and chunked prefill for the burst-64 case.
7. **Cross-scenario confirmation** for a mature winner: run A-D and
   report aggregate geomean speedup.
8. **Block-size 32 with FP8 KV cache** for long-context cases (A, B) once
   FP8 path is validated.

Each round-2 idea should be one PR with one well-defined arm and the
existing round-1 launcher as the comparison baseline.
