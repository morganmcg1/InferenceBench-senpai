# SENPAI Research State — InferenceBench

- **Last updated:** 2026-05-23 10:46 UTC (post round-1 slot-1 cut, baselines retransferred)
- **Most recent research direction from human researcher team:** (none yet for
  this launch — boot only)
- **Research tag:** `ib-20260523-rerun-r1`
- **Advisor branch:** `ib-20260523-rerun-r1-advisor`
- **Hardware (shakedown):** 1 x NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM
- **Wall-clock budget:** 2 hours total

## Current research focus and themes

The research goal is to discover inference-serving configurations and systems
recipes for `mistralai/Mistral-7B-Instruct-v0.3` that beat the default vLLM
launcher's PyTorch-relative speedup on each scenario, while preserving the
MMLU-Pro quality gate and integrity rules. All four scenarios are open for
improvement:

- **A: Input-heavy (TTFT p50, burst conc 1)** — prefill latency on 8192-token
  prompts. Levers: chunked prefill size, max-num-batched-tokens, attention
  backend (FlashInfer/Flash/Triton), CUDA graphs, prefix caching off (concurrency
  1 so likely zero hit), eager-vs-compiled.
- **B: Output-heavy (TPOT p50, burst conc 1)** — long decode on 8192-token
  generations. Levers: speculative decoding (n-gram), FP8 KV cache, chunked
  prefill, kv-cache dtype, attention backend, gpu-memory-utilization.
- **C: High-load (req/s geomean across burst, poisson, constant)** — throughput
  on 256 concurrent requests. Levers: max-num-seqs, max-num-batched-tokens,
  prefix caching, kv-cache fp8, scheduler policy, GPU memory utilization.
- **D: General (geomean TTFT/TPOT/throughput, burst conc 4)** — a balanced
  serving profile. Reusable launchers from A/B/C are good candidates.

## Strategy for this 2-hour window

- 1 GPU shared across 3 students. Serialize heavy benchmark runs to keep
  measurements clean. Students working off the GPU prepare launchers, study
  metrics, or analyze prior runs.
- Each student owns a single scenario for their first PR, focused on the
  highest-leverage knobs. Coordinated GPU slot order in PR bodies.
- All terminal results must come from a clean relaunch + full `evaluate.py`
  run, not from the live training shell.

## Round 1 assignments (revised after slot-1 cut at 10:46 UTC)

First triage at 10:11 UTC: PyTorch baseline metrics + MMLU-Pro registry
absent, so PR #20 was repurposed from Sc C candidate to baseline tooling.

Second triage at 10:46 UTC: r1-frieren went silent after `GPU-CLAIM` at
10:17. r1-tanjiro confirmed GPU at 0 MiB at 10:31 (torch server never
started). r1-frieren did not respond to a 10:30 status ping or a 10:39
escalation with three unblocking options. The 10:43 UTC hard deadline
passed without any progress signal, so the slot was cut and the baseline
build transferred. Closed PR #20 as no-result.

- **r1-tanjiro PR #23 (slot 1) — redirected to baselines.** Run
  `src.eval.inference.precompute_all_baselines --cache-quality-samples
  --backends torch --scenarios B A` with `INFERENCE_BENCH_ALLOW_HF_DOWNLOAD=1`
  (lets the script sample requests internally, skipping the tokenizer
  roundtrip bug r1-frieren hit). Order is **B → A** so r1-fern unblocks
  first. Soft cap 40 min from torch server load. Required minimum: Sc B
  speed baseline + quality registry. Stretch: Sc A speed baseline. r1-tanjiro's
  Sc A candidate launcher (FlashInfer + FP8 + bigbatch) is preserved on
  the branch at `senpai/launchers/A/flashinfer-fp8-bigbatch/start_server.sh`
  (commit `f3c452d`) for round 2.
- **r1-fern PR #22 (slot 2) — unchanged hypothesis, holding.** vLLM
  n-gram speculative decoding + FP8 KV on Scenario B. Waits for
  r1-tanjiro's `SLOT-FREE` + Sc B baseline path. Time-budget contingency:
  if SLOT-FREE lands with <20 min remaining, drop the `--speculative-config`
  flag and run vLLM + FP8 KV only so we land a measurable speedup; if
  <10 min, mark `terminal=false, status=blocked-on-time`.
- **r1-frieren PR #20 — closed, no result.** Sc C aggressive batching
  idea preserved in `research/RESEARCH_IDEAS_2026-05-23_round2.md`.
- **Slot 3 dropped for round 1.** r1-tanjiro's original Sc A candidate
  work is deferred to round 2.

GPU coordination protocol unchanged: each student posts `GPU-CLAIM: <slot>`
before launching the server and `SLOT-FREE: <name> done with GPU` after
teardown.

Round-2 hypothesis catalog: `research/RESEARCH_IDEAS_2026-05-23_round2.md`
(12 ideas across all 4 scenarios — EAGLE-3 spec decode, SGLang RadixAttention,
TGI tuned, TensorRT-LLM, AWQ / FP8-weights quantization, attention backend
ablations, cross-scenario confirmation). Carryover from round 1 to round 2:
r1-frieren's Sc C aggressive batching idea, r1-tanjiro's Sc A FlashInfer + FP8
+ bigbatch launcher (already committed on branch `r1-tanjiro/scenario-a-flashinfer-fp8`).
Top wave once baselines land: H2 (EAGLE-3 on Sc B) and H6 (Sc D chunked prefill).

## Potential next research directions

- **SGLang as alternative engine** on whichever scenario shows the weakest vLLM
  result (especially Scenario A/D where SGLang's RadixAttention can help, but
  burst concurrency 1 may erase that).
- **TGI** as a fallback engine for stability.
- **Speculative decoding variants:** longer `num_speculative_tokens` for
  Scenario B; EAGLE / Medusa if vLLM supports a path from CLI.
- **Quantization sweep:** GPTQ / AWQ / FP8 weights — only if quality gate still
  passes.
- **Attention backend ablation:** FLASH_ATTN vs FLASHINFER vs TRITON_ATTN per
  scenario.
- **Scheduler policy ablation:** `--scheduling-policy` priority vs fcfs for
  Scenario C poisson/constant traffic.
- **Cross-scenario confirmation runs** for mature winners — once a launcher
  beats baseline on one scenario, sanity-check A-D for the geomean leaderboard.
- **Compile/graph capture caveats** on RTX PRO 6000 Blackwell — measure
  warm-up cost before declaring CUDA graph wins.
