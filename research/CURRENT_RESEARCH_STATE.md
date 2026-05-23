# SENPAI Research State

- **Snapshot:** 2026-05-23 (advisor boot for `ib-20260523-rerun-r5`)
- **Most recent human directive:** none — no open GitHub Issues addressed to
  `ib-20260523-rerun-r5-advisor` or the whole team.

## Current Focus

This is a fresh 2-hour shakedown launch with **no PyTorch baselines on disk**
yet. `senpai/preflight.py` fails for all four scenarios on entry. With three
students sharing one RTX PRO 6000 Blackwell GPU, the round is deliberately
narrowed to **Scenario C (high load / request throughput)** so we can get at
least one terminal `SENPAI-RESULT` end-to-end against a real on-disk torch
baseline before the wall-clock cutoff.

Scenario C is the highest-leverage scenario in the public reference snapshot
(46.7x SMAC3 vs PyTorch baseline) and uses short prompts, so its torch
baseline is the cheapest to bootstrap and its full eval is the fastest to
turn around. Scenarios A, B, D are intentionally deferred to a follow-up
round.

## In-Flight Assignments (Round 1, ib-20260523-rerun-r5)

| Student | PR | Role | Hypothesis | GPU order |
|---|---|---|---|---|
| r5-frieren | #24 | Tooling | Bootstrap Scenario C PyTorch torch baseline + MMLU-Pro quality registry | 1st |
| r5-fern | #25 | Launcher v1 | vLLM Scenario C throughput launcher, FP16 KV cache, max-num-seqs=256, mnb-tokens=16384, prefix-caching off, chunked-prefill on, gpu-util=0.92 | 2nd |
| r5-tanjiro | #27 | Launcher v2 | vLLM Scenario C launcher with FP8 KV cache + max-num-seqs=512 + FlashInfer attention backend; tests whether FP8 KV admits more concurrent decode | 3rd |

GPU coordination: r5-frieren owns the GPU exclusively until they post
`SLOT-FREE` on PR #24. r5-fern then runs; posts `SLOT-FREE` on PR #25. Then
r5-tanjiro runs. r5-fern and r5-tanjiro do Phase 1 (launcher authoring,
non-GPU) in parallel with r5-frieren's bootstrap.

## Potential Next-Round Directions

Listed for future advisor invocations once the Scenario C shakedown closes:

1. **Bootstrap Scenarios A, B, D torch baselines** so the other three primary
   metrics are reachable. Sequence by smallest wall-clock first: A (128 reqs,
   long-input), D (96 reqs, balanced), B (64 reqs, long-output) — although
   B's per-request latency is highest.
2. **Scenario A launcher** focused on long-context prefill latency: increase
   `max-num-batched-tokens` substantially, FlashAttention backend, possibly
   speculative decoding (`--speculative-config`) with n-gram speculators.
3. **Scenario B launcher** focused on long decode: CUDA-graph-friendly batch
   sizes, larger `max-num-seqs`, evaluate speculative decoding which can have
   the largest impact on decode-bound workloads.
4. **Mature Scenario C winner cross-scenario confirmation**: run the same
   recipe against A/B/D to compute `aggregate/geomean_speedup_over_pytorch`
   matching the leaderboard.
5. **TensorRT-LLM or SGLang** alternatives once vLLM baselines are well
   characterized, to broaden the engine search space.
6. **Speculative decoding** (`num_speculative_tokens` 3/5/7) is the highest-EV
   single lever on decode-bound workloads (Scenario B, partial Scenario C),
   currently unused in either fern or tanjiro's launchers.
7. **Block size / attention backend ablation** (Triton vs FlashInfer vs
   FlashAttention) once a stable baseline launcher exists.
8. **Quality-gate boundary check** — pick the most aggressive throughput
   launcher and confirm it survives the MMLU-Pro tau=0.95 cutoff before
   committing to it.

## Open Risks

- Torch baseline bootstrap on Scenario C may exceed 75 minutes on RTX PRO 6000;
  if so we have to pivot to a cheaper bootstrap shape (smaller MMLU-Pro N,
  fewer profiles) and re-issue assignments.
- FlashInfer JIT can stall on Blackwell on first use; tanjiro's launcher has a
  documented fallback to the default backend.
- 2-hour budget is tight. Quick-eval-only fallback is allowed for the launcher
  PRs if full eval doesn't fit, but it must be flagged as shakedown evidence
  rather than a leaderboard claim.
