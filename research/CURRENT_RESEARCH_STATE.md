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

## Round 1 timing snapshot (updated 2026-05-23 11:33 UTC — DECISION POINT)

**Bootstrap is much more expensive than originally planned.** r5-frieren's
11:30 update revealed Scenario C bootstrap has 3 speed profiles
(burst + poisson + constant) × 256 reqs = 768 total speed-precompute reqs,
plus MMLU-Pro 500-question quality eval. Realized pace ~4.9 req/min
sequential torch (concurrency=1). Revised full SLOT-FREE ETA:
**~13:15-13:45 UTC**, ~75-105 min past the 12:00 budget edge.

**ADVISOR DECISION at 11:32 UTC:** Let r5-frieren complete the full
bootstrap rather than kill-and-pivot to a burst-only baseline. Reasoning:

1. precompute_all_baselines.py walks profiles via the protected scenario
   JSON; there is no clean burst-only override. A hacked partial baseline
   would be worse than no baseline.
2. r5-fern (PR #25) and r5-tanjiro (PR #27) Phase 1 launcher commits are
   already durable artifacts on their branches. They carry forward to the
   next launch.
3. A complete torch baseline is the prerequisite for every future vLLM PR.
   Killing now wastes the 90 min already invested.

**Consequence:** r5-fern and r5-tanjiro will not run vLLM in this launch.
They've been told to stand down, post a 'Phase 1 complete' marker after gh
rate limit resets at 12:00 UTC, and let the next launch pick up their
launchers against r5-frieren's complete baseline.

- **PR #24 (r5-frieren)**: Speed precompute mid-flight. Burst done 256/256,
  poisson 120/256, constant pending, then MMLU-Pro. ETA SLOT-FREE 13:15-13:45.
- **PR #25 (r5-fern)**: Stand-down. Phase 1 launcher (2e5fa1b) preserved.
- **PR #27 (r5-tanjiro)**: Stand-down. Phase 1 launcher (184c2ef) preserved.
  Container-def vLLM deps bug is a separate follow-up tooling PR for them
  to file after the launch closes.

## Operational: gh rate limit (observed 11:26-11:28 UTC)

Student group pod is hitting GitHub API rate limit (user ID 20516801, HTTP
403). Affects students' ability to poll PR comments and post SLOT-FREE.
Hourly limit resets at 12:00 UTC. GPU work is unaffected (git-local), so
bootstrap continues. Effect: SLOT-FREE may post late and r5-fern may not
read advisor comments promptly. Avoid spamming PR comments while
rate-limited — they consume the students' own quota when polled.

## Open Risks

- Torch baseline bootstrap on Scenario C may exceed 75 minutes on RTX PRO 6000;
  current pace of 48 reqs in 12 min indicates ~64 min for the full 256, plus
  MMLU-Pro. We are inside the 75-min envelope but not by much.
- FlashInfer JIT can stall on Blackwell on first use; tanjiro's launcher has a
  documented fallback to the default backend.
- 2-hour budget is tight. Quick-eval-only fallback is allowed for the launcher
  PRs if full eval doesn't fit, but it must be flagged as shakedown evidence
  rather than a leaderboard claim.
- vLLM container deps bug (uv install with `--no-deps` then `2>/dev/null || true`
  on the resolving install) was hit and worked around locally by r5-tanjiro;
  needs a hardening PR once the round closes.
