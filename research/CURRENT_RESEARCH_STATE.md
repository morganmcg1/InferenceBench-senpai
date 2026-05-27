# SENPAI Research State

- **Timestamp:** 2026-05-27 17:19 UTC (~55 min remaining in 2 h window)
- **Most recent direction from human researcher team:** none (no open issues from the human team; I filed ops issue #130 at 17:09 to surface the silent-student situation).
- **Run setup:**
  - Tag: `ib-20260527-lean1-r1`
  - 3 students (frieren, fern, tanjiro) sharing 1 RTX PRO 6000.
  - W&B project `wandb-applied-ai-team/inferencebench-senpai`.

## Current best metrics on this hardware

| Scenario | Speedup | PR | W&B | Status |
|---|---:|---|---|---|
| A | **1.893x** | #126 (merged 17:17) | u44zjwyh | TTFT p50=0.232s; quality=1.0; 128/128 |
| B | _unset_ | #131 (assigned to frieren, quick probe only) | — | frieren running n-gram spec decoding probe |
| C | **25.62x** | #128 (merged) | ckfmuinz | first row; tanjiro #129 stalled |
| D | _unset_ | _unassigned_ | — | not on the slate this run |

## Active work

- **#131 frieren Scenario B n-gram spec decoding** — quick probe only (HARD LIMIT: no full eval, time constraint). Will establish first Scenario B measurement on RTX PRO 6000. Assigned 17:19.
- **#129 tanjiro Scenario C multi-step scheduler** — tanjiro container silent since 16:50:17; no work started.
- **#127 fern Scenario B n-gram spec** — fern container silent since 16:22:24; no work started.

## Operational issues

- **fern container silent** since iter 16 at 16:22:24 UTC (~46 min). PR #127 left open `status:wip` in case orchestrator resumes worker.
- **tanjiro container silent** since iter 22 at 16:50:17 UTC (~18 min). PR #129 left open `status:wip`.
- **Advisor SA lacks pods/exec** — cannot inspect or restart per-student claude processes. Filed issue #130 asking the human research team to nudge the pod if revival is feasible before 18:00 UTC. If both containers stay dead, scenarios B and D end the launch with no measurement; only A may join C in the BASELINE ledger.

## Remaining decision points (sequenced by clock)

1. When frieren's #131 quick probe lands (~17:35-17:45): validate the result and merge the PR as a research signal — no full eval needed.
2. If fern or tanjiro container revives before 17:50: a quick probe is feasible; direct them to their current open PR (fern → #127 Scenario B base, tanjiro → #129 Scenario C multi-step). No full eval.
3. **Hard stop at 17:50** — do not start any new evaluation of any kind after 17:50 UTC.
4. 17:50-18:14: finalize, commit final state, ensure all advisors docs are current.

## Lessons being banked

- FP8 weights are quality-safe at n=500 on Mistral-7B-Instruct-v0.3 (proven on Scenario C; reasonable to assume on A and B).
- Quick-mode quality is unreliable below n~30 (16-sample MMLU subset gives ratio 0.839 for any decent server). Don't act on quick quality.
- Concurrency=1 scenarios (A: 128×8K input, B: 64×8K output) take 40-50 min for full eval — a major time sink. Quick-then-full pipeline must include this in scheduling.
- vLLM on RTX PRO 6000 boots cleanly with FP8 weights + BF16 KV cache (FP8 KV remains FlashAttention-incompatible). FlashInfer disabled via `runtime_env.sh` keeps things stable.
