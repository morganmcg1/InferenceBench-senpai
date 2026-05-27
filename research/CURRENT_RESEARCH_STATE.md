# SENPAI Research State

- **Timestamp:** 2026-05-27 17:43 UTC (~31 min remaining in 2 h window)
- **Most recent direction from human researcher team:** none (ops issue #130 has no response from human team yet).
- **Run setup:**
  - Tag: `ib-20260527-lean1-r1`
  - 3 students (frieren, fern, tanjiro) sharing 1 RTX PRO 6000.
  - W&B project `wandb-applied-ai-team/inferencebench-senpai`.

## Current best metrics on this hardware

| Scenario | Speedup | PR | W&B | Status |
|---|---:|---|---|---|
| A | **1.893x** | #126 (merged 17:17) | u44zjwyh | TTFT p50=0.232s; quality=1.0; 128/128 |
| B | _unset_ (research signal: 3.5x quick replicated, screening only) | #127 (closed), #131 (closed) | 5b0w8j17, 8bwybtey | both students 3.5x — recipe ready for next launch full eval |
| C | **25.62x** | #128 (merged); #129 (tanjiro full eval ran ~17:25-17:38, result pending) | ckfmuinz | tanjiro may have full result coming on async-scheduling recipe |
| D | _unset_ | #132 (fern quick probe in flight, ~17:31 start) | pending | first D measurement attempt |

## Active work

- **#129 tanjiro Scenario C async-scheduling** — tanjiro revived ~17:18, posted 3.95x quick at 17:21, then ran full eval ~17:25-17:38 (per frieren's GPU-slot notes). Full SENPAI-RESULT not posted yet. **No launcher commit yet** — pinged tanjiro at 17:42 to push.
- **#132 fern Scenario D fp8-chunked-prefill** — first D measurement quick probe. In flight ~17:31; landed by ~17:50.
- **frieren**: idle, no new assignment (8 min to hard stop, can't run anything new).

## Closed this round

- **#127 fern Scenario B n-gram-spec** (closed 17:28) — 3.51x research signal.
- **#131 frieren Scenario B n-gram-spec** (closed 17:42) — 3.475x research signal, independent replication.

## Research signals banked this launch (awaiting next launch's full evals)

- **Scenario B (BF16 + n-gram k=5)**: 3.51x TPOT speedup quick (n=4 speed, n=16 quality). FP8 weights dominated here — drop FP8, use BF16 + ngram. Recommend full eval with gpu-mem-util=0.75 next launch.
- **Scenario C (async-scheduling over #128 recipe)**: 3.95x quick geomean. The 2.76x→25.62x amplification precedent from #128 suggests full eval could be significantly higher. High priority for next launch.

## Key lessons banked

- FP8 weights: quality-safe on A (ratio=1.0) and C (ratio=1.0); quality-risky on B (0.629 screening, dominates no speed gain). Do NOT use FP8 on decode-heavy workloads.
- N-gram spec decoding: `--speculative-config '{"method":"ngram",...}'` JSON form works in vLLM 0.11.0 v1 on RTX PRO 6000. 3.5x TPOT at c=1 output-heavy.
- `--num-scheduler-steps` was removed in vLLM v1. Use `--async-scheduling` instead on v1.
- VRAM 88-90 GiB at gpu-mem-util=0.90 exceeds H100 80 GB envelope — reduce to 0.75 for portability.

## Operational issues

- **fern container silent** since iter 16 at 16:22:24 UTC (~46 min). PR #127 left open `status:wip` in case orchestrator resumes worker.
- **tanjiro container silent** since iter 22 at 16:50:17 UTC (~18 min). PR #129 left open `status:wip`.
- **Advisor SA lacks pods/exec** — cannot inspect or restart per-student claude processes. Filed issue #130 asking the human research team to nudge the pod if revival is feasible before 18:00 UTC. If both containers stay dead, scenarios B and D end the launch with no measurement; only A may join C in the BASELINE ledger.

## Remaining decision points (sequenced by clock)

1. **~17:35-17:45**: #131 (frieren B quick) and #132 (fern D quick) expected to land. Review as research signals.
2. **~17:35-17:45**: #129 (tanjiro C async-scheduling) may post terminal quick. Review as research signal; do NOT merge without terminal quality gate.
3. **Hard stop 17:50**: No new eval may be started after 17:50 UTC.
4. **17:50-18:14**: Finalize all docs, commit updated CURRENT_RESEARCH_STATE and EXPERIMENTS_LOG, ensure BASELINE.md is accurate.

## Lessons being banked

- FP8 weights are quality-safe at n=500 on Mistral-7B-Instruct-v0.3 (proven on Scenario C; reasonable to assume on A and B).
- Quick-mode quality is unreliable below n~30 (16-sample MMLU subset gives ratio 0.839 for any decent server). Don't act on quick quality.
- Concurrency=1 scenarios (A: 128×8K input, B: 64×8K output) take 40-50 min for full eval — a major time sink. Quick-then-full pipeline must include this in scheduling.
- vLLM on RTX PRO 6000 boots cleanly with FP8 weights + BF16 KV cache (FP8 KV remains FlashAttention-incompatible). FlashInfer disabled via `runtime_env.sh` keeps things stable.
