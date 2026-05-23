# SENPAI Research State — InferenceBench

- **Snapshot time:** 2026-05-23 (launch start)
- **Run tag:** `ib-20260523-rerun-r4`
- **Advisor branch:** `ib-20260523-rerun-r4-advisor`
- **Students:** r4-frieren, r4-fern, r4-tanjiro (3 idle at launch)
- **GPUs:** 1 RTX PRO 6000 Blackwell shared across the 3 students in this pod
- **Wall budget:** 2 hour SENPAI window (assignment + eval + review + cleanup)

## Most recent direction from human research team

None. Issue queue is empty at launch.

## Current research focus and themes

1. **Unblock scoring first.** Preflight on RTX PRO 6000 is FAIL across all
   scenarios — torch speed baselines, deterministic requests, MMLU-Pro
   samples, and the torch quality registry are missing. Without those,
   students cannot compute paper-facing `speedup_over_pytorch`. First action
   for this launch: one tooling PR to run
   `python -m src.eval.inference.precompute_all_baselines` and commit the
   resulting artifacts.

2. **Parallel launcher prep on the other two students.** While baseline
   materialization holds the single GPU, the other two students draft
   high-confidence vLLM/SGLang launcher recipes under
   `senpai/launchers/<scenario>/<slug>/start_server.sh` that they will
   smoke-test the moment the GPU frees up.

3. **Scenario priorities for first measured arm:**
   - **B (output-heavy / TPOT):** highest historical headroom (public H100
     SMAC3 reached 15.23x). Levers: vLLM with FP8 KV cache, FlashInfer
     attention backend, n-gram speculative decoding, careful
     `max_num_batched_tokens`.
   - **C (high-load throughput):** public SGLang default already at 51.12x.
     Levers: high concurrency + chunked prefill + prefix caching.
   - A and D are deferred to round 2 once baselines exist and B/C launchers
     have settled.

4. **One scenario per PR.** Geomean confirmation across A-D is only worth
   running for mature winners or broadly reusable launchers.

## Potential next research directions

- After baselines exist: probe FP8 KV cache + FlashInfer + n-gram speculative
  decoding for B (TPOT).
- After baselines exist: probe SGLang with chunked prefill, prefix caching,
  high `max_running_requests` for C (throughput).
- After B/C settle: open A (long-context TTFT — chunked prefill, attention
  backend, KV-cache layout) and D (balanced — pick best of B/C config and
  tune around the mean workload).
- Cross-scenario geomean confirmation once a single launcher beats baseline
  on multiple scenarios.
- If a vLLM recipe wins on RTX PRO 6000, prepare an H100 reproduction PR for
  paper-facing leaderboard claim.

## Coordination notes

- Heavy GPU work must serialize on the single benchmark GPU. The advisor
  steers which student owns the GPU at each moment via PR comments.
- Students should post a `SLOT-FREE` signal when their server process group
  is torn down so the next student can launch cleanly. They must not use
  broad `pkill` — only kill their own process group.
- Treat `SENPAI_TIMEOUT_MINUTES` and `SENPAI_MAX_EPOCHS` as hard per-run
  bounds.
