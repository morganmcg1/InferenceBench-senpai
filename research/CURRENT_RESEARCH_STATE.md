# SENPAI Research State

- **Timestamp:** 2026-05-27 17:00 UTC (~74 min remaining in 2 h window)
- **Most recent direction from human researcher team:** none (no open issues).
- **Run setup:**
  - Tag: `ib-20260527-lean1-r1`
  - 3 students (frieren, fern, tanjiro) sharing 1 RTX PRO 6000.
  - W&B project `wandb-applied-ai-team/inferencebench-senpai`.

## Current best metrics on this hardware

| Scenario | Speedup | PR | W&B | Status |
|---|---:|---|---|---|
| A | _unset_ | #126 (in flight) | u2dxy31b (quick=1.91x) | frieren full eval running ~16:46–17:35 |
| B | _unset_ | #127 (stalled) | — | fern pod has not iterated since 16:22:24, PR open but inert |
| C | **25.62x** | #128 (merged) | ckfmuinz | first row landed; tanjiro #129 chasing higher |
| D | _unset_ | _unassigned_ | — | not yet on the slate this run |

## Round 2 in progress

- **#129 tanjiro Scenario C multi-step scheduler** — adds `--num-scheduler-steps 8` over the #128 winner recipe; targets 25.62x to improve. Tanjiro will be GPU-queued behind frieren.
- **#126 frieren Scenario A full eval** — chunked prefill + FP8 weights; quick was 1.91x; full result expected ~17:35.

## Operational issues

- **fern pod silent** — senpai-fern container last heartbeat at 16:23, iter 16 in progress. No new iterations for ~37 minutes. PR #127 left open with `status:wip` in case the orchestrator resumes the worker; no further chasing.

## Remaining decision points (sequenced by clock)

1. When frieren's #126 full eval lands (~17:35): validate, finalize, and merge if it improves over PyTorch baseline on Scenario A. Update BASELINE.md row A.
2. After #129 quick probe lands (likely ~17:40 if tanjiro can grab GPU after frieren): decide on full eval. If quick clearly beats 25.62x quick-stage equivalent and quality plausible, promote to full eval (~10 min). Hard stop at 18:00 to leave 14 min for finalization.
3. If time permits between landings, queue tanjiro a Scenario A or B follow-up using insights from frieren's full result.
4. Do NOT start any new full eval after 17:50 — too tight for review/merge before the 18:14 cutoff.

## Lessons being banked

- FP8 weights are quality-safe at n=500 on Mistral-7B-Instruct-v0.3 (proven on Scenario C; reasonable to assume on A and B).
- Quick-mode quality is unreliable below n~30 (16-sample MMLU subset gives ratio 0.839 for any decent server). Don't act on quick quality.
- Concurrency=1 scenarios (A: 128×8K input, B: 64×8K output) take 40-50 min for full eval — a major time sink. Quick-then-full pipeline must include this in scheduling.
- vLLM on RTX PRO 6000 boots cleanly with FP8 weights + BF16 KV cache (FP8 KV remains FlashAttention-incompatible). FlashInfer disabled via `runtime_env.sh` keeps things stable.
