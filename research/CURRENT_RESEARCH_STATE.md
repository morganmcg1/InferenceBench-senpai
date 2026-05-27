# SENPAI Research State

- **Timestamp:** 2026-05-27 17:10 UTC (~64 min remaining in 2 h window)
- **Most recent direction from human researcher team:** none (no open issues from the human team; I filed ops issue #130 at 17:09 to surface the silent-student situation).
- **Run setup:**
  - Tag: `ib-20260527-lean1-r1`
  - 3 students (frieren, fern, tanjiro) sharing 1 RTX PRO 6000.
  - W&B project `wandb-applied-ai-team/inferencebench-senpai`.

## Current best metrics on this hardware

| Scenario | Speedup | PR | W&B | Status |
|---|---:|---|---|---|
| A | _unset_ | #126 (in flight) | u2dxy31b (quick=1.91x) | frieren full eval running, expected ~17:35 |
| B | _unset_ | #127 (stalled) | — | fern container last heartbeat 16:22:24 (iter 16); silent ~46 min |
| C | **25.62x** | #128 (merged) | ckfmuinz | first row landed; tanjiro #129 chasing higher but stalled |
| D | _unset_ | _unassigned_ | — | not on the slate this run |

## Round 2 status

- **#126 frieren Scenario A full eval** — alive, GPU 96% / 90 GiB allocated; iter 24 at 17:08:04 UTC. Full result expected ~17:35. Only active student.
- **#129 tanjiro Scenario C multi-step scheduler** — assigned 16:50; tanjiro container fell silent at iter 22 at 16:50:17 UTC (last heartbeat). No work begun.

## Operational issues

- **fern container silent** since iter 16 at 16:22:24 UTC (~46 min). PR #127 left open `status:wip` in case orchestrator resumes worker.
- **tanjiro container silent** since iter 22 at 16:50:17 UTC (~18 min). PR #129 left open `status:wip`.
- **Advisor SA lacks pods/exec** — cannot inspect or restart per-student claude processes. Filed issue #130 asking the human research team to nudge the pod if revival is feasible before 18:00 UTC. If both containers stay dead, scenarios B and D end the launch with no measurement; only A may join C in the BASELINE ledger.

## Remaining decision points (sequenced by clock)

1. When frieren's #126 full eval lands (~17:35): validate, finalize, and merge if it improves over PyTorch baseline on Scenario A. Update BASELINE.md row A.
2. If fern or tanjiro container revives, they only have time for a quick probe at best — no full eval after 17:50.
3. Do NOT start any new full eval after 17:50 — too tight for review/merge before the 18:14 cutoff.

## Lessons being banked

- FP8 weights are quality-safe at n=500 on Mistral-7B-Instruct-v0.3 (proven on Scenario C; reasonable to assume on A and B).
- Quick-mode quality is unreliable below n~30 (16-sample MMLU subset gives ratio 0.839 for any decent server). Don't act on quick quality.
- Concurrency=1 scenarios (A: 128×8K input, B: 64×8K output) take 40-50 min for full eval — a major time sink. Quick-then-full pipeline must include this in scheduling.
- vLLM on RTX PRO 6000 boots cleanly with FP8 weights + BF16 KV cache (FP8 KV remains FlashAttention-incompatible). FlashInfer disabled via `runtime_env.sh` keeps things stable.
