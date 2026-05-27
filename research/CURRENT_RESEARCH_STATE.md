# SENPAI Research State

- **Timestamp:** 2026-05-27 18:10 UTC (launch ends 18:14; all students idle; no new assignments — T-3.6 min is past T-24 min cutoff for new evals per ops playbook)
- **Most recent direction from human researcher team:** none (ops issue #130 has no response from human team yet).
- **Run setup:**
  - Tag: `ib-20260527-lean1-r1`
  - 3 students (frieren, fern, tanjiro) sharing 1 RTX PRO 6000.
  - W&B project `wandb-applied-ai-team/inferencebench-senpai`.

## Current best metrics on this hardware

| Scenario | Speedup | PR | W&B | Status |
|---|---:|---|---|---|
| A | **1.893x** | #126 (merged 17:17) | u44zjwyh | TTFT p50=0.232s; quality=1.0; 128/128 |
| B | _unset_ (research signal: 3.5x quick replicated) | #127 (closed), #131 (closed) | 5b0w8j17, 8bwybtey | both students 3.5x — recipe ready for next launch |
| C | **27.24x** | #128 (merged); #129 (MERGED 18:03) | ckfmuinz / xzd8kbha | async-scheduling +6.3% over #128. quality_ratio=0.973 n=500. |
| D | _unset_ (research signal: 1.44x quick) | #132 (closed 17:48) | qeo5rbof | first D measurement; recipe ready for next launch |

## Active work

- All students idle. No PRs open. Launch closed at 18:04 UTC.

## Closed this round

- **#127 fern Scenario B n-gram-spec** (closed 17:28) — 3.51x research signal.
- **#131 frieren Scenario B n-gram-spec** (closed 17:42) — 3.475x research signal, independent replication.
- **#132 fern Scenario D FP8 + chunked-prefill** (closed 17:48) — 1.44x first D measurement research signal.
- **#129 tanjiro Scenario C async-scheduling** (MERGED 18:03) — 27.24x terminal full eval. New Scenario C baseline.

## Launch summary (running tally)

- **3 merged baseline winners** this launch: PR #126 Scenario A 1.893x, PR #128 Scenario C 25.62x, PR #129 Scenario C 27.24x (new best).
- **2 research-signal-grade quick probes** banked: B 3.5x (replicated × 2), D 1.44x.
- **Universal lesson**: FP8 + chunked-prefill works on A/C/D; BF16 + n-gram-spec works on B. For C specifically, add `--async-scheduling` on top of chunked-prefill + prefix-caching. Clean playbook for next launch.

## Research signals banked this launch (awaiting next launch's full evals)

- **Scenario B (BF16 + n-gram k=5)**: 3.51x TPOT speedup quick (n=4 speed, n=16 quality). FP8 weights dominated here — drop FP8, use BF16 + ngram. Recommend full eval with gpu-mem-util=0.75 next launch.
- **Scenario C (async-scheduling)**: **CONFIRMED BASELINE 27.24x** (PR #129 merged). Full eval 768/768 speed requests, quality_ratio=0.973 (n=500), W&B xzd8kbha. The `--async-scheduling` flag gives measurable throughput gain on poisson/constant profiles. Next levers: block-size 32 sweep, further gpu-mem-util tuning for H100.

## Key lessons banked

- FP8 weights: quality-safe on A (ratio=1.0) and C (ratio=1.0); quality-risky on B (0.629 screening, dominates no speed gain). Do NOT use FP8 on decode-heavy workloads.
- N-gram spec decoding: `--speculative-config '{"method":"ngram",...}'` JSON form works in vLLM 0.11.0 v1 on RTX PRO 6000. 3.5x TPOT at c=1 output-heavy.
- `--num-scheduler-steps` was removed in vLLM v1. Use `--async-scheduling` instead on v1.
- VRAM 88-90 GiB at gpu-mem-util=0.90 exceeds H100 80 GB envelope — reduce to 0.75 for portability.

## Operational issues (resolved or accepted)

- **fern container** went silent 16:22-17:18 (~56 min), revived ~17:18, posted #127 result at 17:18 and 17:26, took new #132 assignment and landed D result at 17:47. **Worked out fine** — orchestrator restored worker without human intervention.
- **tanjiro container** went silent 16:50-17:18 (briefly), revived ~17:18 to post #129 quick result and run full eval, then silent again. Never pushed launcher commit or terminal SENPAI-RESULT. **Partial recovery only** — research signal preserved in comment but recipe not committed.
- **Issue #130** filed at 17:09 about silent containers; human team did not respond (0 comments) but both containers self-revived after the issue was filed. Will leave issue open for next-launch diagnostics review.
- **Advisor SA lacks pods/exec** — known limitation; cannot inspect or restart per-student processes from advisor side.

## Remaining decision points (sequenced by clock)

1. **17:48 → 18:14**: All probes done. Only outstanding work is #129 — if tanjiro pushes a terminal result with a measurable improvement over 25.62x and the quality gate passes at n=500, merge as new Scenario C winner. Otherwise close as research signal.
2. **18:00 cutoff for merge attempts** — leave 14 min buffer for any merge conflicts or BASELINE.md updates.
3. **17:48-18:14**: Finalize all docs (this file, EXPERIMENTS_LOG.md, BASELINE.md). Ensure next launch's playbook is captured.

## Next-launch playbook (banked from this round)

**Scenario A (current best 1.893x, headroom to H100 reference 4.37x)**:
- AWQ/INT4 weight quantization over the #126 winner — biggest remaining lever for TTFT.
- Speculative decoding probe (likely small benefit since TTFT is prefill-bound).

**Scenario B (no measured baseline, research signal 3.5x BF16+ngram)**:
- **First priority**: full eval of `B/ngram-spec/start_server.sh` (BF16 + ngram k=5, gpu-mem-util=0.75) to land first B baseline.
- `num_speculative_tokens` sweep {3,5,7,9} after base lands.
- EAGLE-style draft model speculation.

**Scenario C (current best 27.24x from #129, headroom unclear)**:
- **First priority**: `--block-size 32` vs default 16 for cache locality at c=64 on top of the #129 baseline.
- Sweep `--max-num-seqs` {128, 256, 512} at high concurrency.
- H100 portability: reduce gpu-mem-util 0.92 → 0.75 and confirm throughput holds.

**Scenario D (no measured baseline, research signal 1.44x FP8+chunked-prefill)**:
- **First priority**: full eval of `D/fp8-chunked-prefill/start_server.sh` to land first D baseline.
- Add `--enable-prefix-caching` and `--max-num-seqs` sweep.

**Cross-scenario hardening**:
- Drop gpu-memory-utilization 0.90 → 0.75 for H100 80GB portability on all launchers.

## Lessons being banked

- FP8 weights are quality-safe at n=500 on Mistral-7B-Instruct-v0.3 (proven on Scenario C; reasonable to assume on A and B).
- Quick-mode quality is unreliable below n~30 (16-sample MMLU subset gives ratio 0.839 for any decent server). Don't act on quick quality.
- Concurrency=1 scenarios (A: 128×8K input, B: 64×8K output) take 40-50 min for full eval — a major time sink. Quick-then-full pipeline must include this in scheduling.
- vLLM on RTX PRO 6000 boots cleanly with FP8 weights + BF16 KV cache (FP8 KV remains FlashAttention-incompatible). FlashInfer disabled via `runtime_env.sh` keeps things stable.
