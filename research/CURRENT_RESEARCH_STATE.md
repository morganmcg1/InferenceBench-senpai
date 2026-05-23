# SENPAI Research State — InferenceBench

- **Last updated:** 2026-05-23 11:21 UTC (option-R pivot: kill baselines, r1-tanjiro takes partial Sc A)
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

## Round 1 assignments (post 11:21 UTC option-R pivot)

Hard cluster cutoff verified at **11:48:08 UTC** (start gate file at
`/mnt/new-pvc/senpai-start-gates/ib-20260523-rerun/start`); harvest
starts at 11:43:08. r1-frieren's first precompute died at 10:49 (process
death after the session boundary; no baseline written). They restarted
at 11:13:57 with `setsid` + nohup-style detach. Revised projection
landed Sc A baseline at ~11:48 — right at cutoff with zero buffer for
commit+push, and quality registry + SLOT-FREE all past cutoff. Advisor
pivoted to **option R**: kill the baseline run, accept no torch baseline,
let r1-tanjiro run a partial Sc A vLLM candidate with raw TTFT only.

- **r1-frieren PR #20 — kill directed at 11:21 UTC.** Their round-1
  deliverable is the senpai-only `robust_truncate_messages` patch in
  `senpai/materialize_requests.py` (commit `999179b`) + the materialized
  request files for scenarios A, B, D under
  `src/eval/inference/baselines/speed/torch/<scenario>/<safe_model>/requests.jsonl`.
  These unblock all future torch baseline builds. Required post: `SLOT-FREE`
  after SIGTERM of PID 74981.
- **r1-tanjiro PR #23 — partial Sc A candidate (no baseline, no quality
  registry).** As soon as r1-frieren's SLOT-FREE lands, run the FlashInfer
  + FP8 + bigbatch launcher against the pre-materialized Sc A requests.
  Report raw `1/ttft.p50` and observed MMLU-Pro accuracy, NOT
  `speedup_over_pytorch` (no denominator). Must commit + push by 11:43 UTC
  harvest start. Status: `partial-no-baseline`. Launcher needs the
  `--max-model-len 32768` fix r1-tanjiro caught at 11:16 (Mistral-7B
  position embeddings limit).
- **r1-fern PR #22 — stood down at 11:12 UTC.** Posted
  `terminal=false, status=blocked-on-missing-baseline` cleanly. Launcher
  + workspace preserved for round 2. Flagged
  `senpai/summarize_metrics.py::_baseline_primary_value` path-mismatch
  bug for round-2 tooling fix.

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
