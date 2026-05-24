# SENPAI Research State — ib-20260524-leasefix-r4

- **As of:** 2026-05-24 23:20 UTC (40 min remaining in 2 h SENPAI window)
- **Most recent human-team directive:** none yet for this tag.
- **Hardware:** NVIDIA RTX PRO 6000 Blackwell shakedown run (not
  leaderboard-comparable). Single GPU shared across 3 student logical
  workers in one pod.
- **Target:** `morganmcg1/InferenceBench-senpai` — InferenceBench LLM serving
  benchmark, base model Mistral-7B-Instruct-v0.3, scenarios A/B/C/D, MMLU-Pro
  quality gate at tau=0.95.

## Round 1 outcomes so far

| Scenario | Status | Live speedup | PR | W&B |
|---|---|---:|---|---|
| A | merged | **1.246x** | #83 (frieren) | `7dew56hp` |
| B | WIP — recovery needed | — | #84 (fern) | (no run yet) |
| C | merged (quick-only) | **3.89x** | #92 (frieren) | `8kxj9472` |
| D | WIP — re-queued after wrapper fix | — | #86 (tanjiro) | (no run yet) |

Key engineering takeaways:

- **vLLM 0.11.0 V1 chunked-prefill override:** the engine silently sets
  `chunked_prefill_enabled=True` for non-pooling tasks
  (`vllm/engine/arg_utils.py:1548`) regardless of
  `--no-enable-chunked-prefill`. Documented in BASELINE.md.
- **vLLM 0.11.0 ready signal:** does NOT emit `Uvicorn running`. Wrappers
  must poll `curl -sf /v1/models` and the supervised launcher's
  `[supervised] server ready` line.
- **PROBLEM_DIR=target/ stub:** the launch harness sometimes sets a
  relative stub that breaks `source ${PROBLEM_DIR}/senpai/runtime_env.sh`.
  All launchers must probe `/workspace/senpai-<student>/target` and
  `/workspace/senpai/target` as fallbacks.

## Active PRs (round 1 + 2 mix)

- **#83 frieren** — Scenario A merged at 1.246x.
- **#84 fern** — Scenario B WIP. Held GPU 64 min iteration (22:45-23:06Z)
  but produced no W&B run, PR comment, or git push. Urgent advisor comment
  posted requesting they surface `metrics_*.json` from `/tmp/ib-B/task` or
  paste server log. Hard stop 23:55Z.
- **#86 tanjiro** — Scenario D WIP. Identified vLLM ready-signal bug in
  wrapper (was checking for `Uvicorn running`); fixed to poll `/v1/models`.
  Re-queued via `gpu_slot.py run --wait`. Arm1 quick already showed 1.26x.
  Will run full eval with arm1 (defaults) when slot frees up.
- **#92 frieren** — Scenario C merged at 3.89x (quick-only).
- **#94 frieren** — Scenario A round-2: prefix caching on (single-lever
  delta vs #83). Quick-eval only. Queues LAST behind tanjiro and fern.

## Current research focus

Two missing live baselines (Scenarios B and D) are the immediate priority.
Both are pending student delivery within 40 min. Beyond that, the two
merged baselines (A at 1.246x, C at 3.89x) have significant headroom
relative to the H100 SMAC3 references (A: 4.37x, C: 46.7x).

## Round-2 candidates (post round-1 finalization)

1. **Confirm Scenario C** with full eval at n=256 per profile + MMLU n=500
   (quick-only baseline at 3.89x needs validation).
2. **Speculative decoding for B/D** — vLLM n-gram speculative decoding
   directly targets TPOT for long generation. Quality gate must pass.
3. **FP8 KV cache for B/D** — bandwidth-bound decode benefits from smaller
   KV cache. RTX PRO 6000 supports FP8; controlled hardware-path PR with
   strict quality-gate evidence.
4. **Prefix caching for A** — in flight as PR #94.
5. **Block-size sweep (16 vs 32)** once round-1 baselines all complete.
6. **TensorRT-LLM** or **SGLang** as alternative engines, scenario-by-
   scenario, only if vLLM hits a clear ceiling on one scenario.
7. **Scenario C: larger `--max-num-seqs 512`** and/or
   `--max-num-batched-tokens 16384` — the H100 SMAC3 reference reaches
   46.7x; 3.89x leaves a lot of headroom.
8. **Cross-scenario confirmation** for a mature winner: run A-D and report
   aggregate geomean speedup.
9. **Block-size 32 with FP8 KV cache** for long-context cases (A, B) once
   FP8 path is validated.

Each round-2 idea should be one PR with one well-defined arm and the
existing round-1 launcher as the comparison baseline.
