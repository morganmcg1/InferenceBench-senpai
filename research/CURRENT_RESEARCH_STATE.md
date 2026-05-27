# SENPAI Research State

- **Timestamp:** 2026-05-27 17:48 UTC (~26 min remaining in 2 h window)
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
| C | **25.62x** | #128 (merged); #129 (tanjiro WIP, no launcher commit, no terminal result) | ckfmuinz | tanjiro async-scheduling research signal in PR text only |
| D | _unset_ (research signal: 1.44x quick) | #132 (closed 17:48) | qeo5rbof | first D measurement; recipe ready for next launch |

## Active work

- **#129 tanjiro Scenario C async-scheduling** — tanjiro silent since posting quick result at 17:21. Frieren's notes say tanjiro ran a full eval ~17:25-17:38 but no commit or terminal SENPAI-RESULT was pushed. Pinged at 17:42; no response.
- **frieren, fern**: idle. Cannot assign new work — past 17:50 hard stop is imminent.

## Closed this round

- **#127 fern Scenario B n-gram-spec** (closed 17:28) — 3.51x research signal.
- **#131 frieren Scenario B n-gram-spec** (closed 17:42) — 3.475x research signal, independent replication.
- **#132 fern Scenario D FP8 + chunked-prefill** (closed 17:48) — 1.44x first D measurement research signal.

## Launch summary (running tally)

- **2 merged baseline winners** this launch: PR #126 Scenario A 1.893x, PR #128 Scenario C 25.62x.
- **3 research-signal-grade quick probes** banked: B 3.5x (replicated), D 1.44x, C async-scheduling 3.95x (in comment only).
- **Universal lesson**: FP8 + chunked-prefill works on A/C/D; BF16 + n-gram-spec works on B. Clean playbook for next launch.

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

**Scenario C (current best 25.62x, headroom unclear)**:
- **First priority**: full eval of tanjiro's `--async-scheduling` recipe over #128 winner.
- Compare `--block-size 32` vs default 16 for cache locality at c=64.

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
