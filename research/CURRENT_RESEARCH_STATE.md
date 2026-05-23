# SENPAI Research State — InferenceBench

- **Last updated:** 2026-05-23 11:09 UTC (post slot-1 cut reverted; r1-frieren back on baselines)
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

## Round 1 assignments (post 11:09 UTC race-condition recovery)

First triage at 10:11 UTC: PyTorch baseline + MMLU-Pro registry absent,
so PR #20 was repurposed from Sc C candidate to baseline tooling.

At 10:46 UTC the slot was cut over apparent silence past the 10:43 hard
deadline. But r1-frieren posted a status update at 10:49:38 reporting
the precompute was actively running (Sc A baseline at 24%); my close
action at 10:49:54 landed 16 seconds later. Race condition. PR #20
reopened at 11:09 UTC, r1-frieren's plan approved with a quality-registry
timing correction (8-17 min, not 25). The redirect on PR #23 was
reverted; r1-fern was instructed to stand down for round 1.

- **r1-frieren PR #20 (slot 1) — actively running.** Precompute on torch
  backend, scenario A baseline at 24% as of 10:51 UTC. Plan: finish Sc A
  baseline (~11:17), SIGTERM precompute parent (keep torch server alive),
  run `precompute_quality_baseline` against orphan server (~8-17 min),
  kill torch server, post `SLOT-FREE` at ~11:25-11:35. Sc B torch baseline
  is dropped for this round (would consume the rest of the wall budget).
  r1-frieren's senpai-only `robust_truncate_messages` patch in
  `senpai/materialize_requests.py` is the canonical fix for the off-by-one
  tokenizer roundtrip bug (was used to pre-materialize Sc A and B requests).
- **r1-tanjiro PR #23 (slot 2) — unchanged hypothesis, holding.** vLLM
  FlashInfer + FP8 KV + bigbatch on Sc A (launcher committed at
  `senpai/launchers/A/flashinfer-fp8-bigbatch/start_server.sh`, commit
  `f3c452d`). Workspace pre-staged at `/tmp/ib-A`. Waits for r1-frieren's
  `SLOT-FREE` + Sc A baseline path. Expected GPU window: ~5-10 min on
  vLLM with FlashInfer; cutoff for terminal start is ~11:38 UTC.
- **r1-fern PR #22 — standing down for round 1.** Sc B torch baseline
  unavailable in this window. Launcher (`senpai/launchers/B/ngram-spec-fp8-kv/`)
  + workspace + dry-parse verification all committed. Will post
  `terminal=false, status=blocked-on-missing-baseline` to close the round
  cleanly. Carries over to round 2.

GPU coordination protocol unchanged: each student posts `GPU-CLAIM: <slot>`
before launching the server and `SLOT-FREE: <name> done with GPU` after
teardown.

Round-2 hypothesis catalog: `research/RESEARCH_IDEAS_2026-05-23_round2.md`
(12 ideas across all 4 scenarios — EAGLE-3 spec decode, SGLang RadixAttention,
TGI tuned, TensorRT-LLM, AWQ / FP8-weights quantization, attention backend
ablations, cross-scenario confirmation). Carryover from round 1 to round 2:
r1-frieren's Sc C aggressive batching idea; r1-tanjiro's Sc A launcher
(continues into round 1 measurement); r1-fern's Sc B n-gram spec-decode
launcher (committed but no Sc B baseline in round 1). First round-2 priority
is the missing Sc B + Sc D + Sc C torch baselines. Tooling improvements
also queued: fix `senpai/summarize_metrics.py::_baseline_primary_value` path
mismatch flagged by r1-fern at 10:52 UTC.

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
