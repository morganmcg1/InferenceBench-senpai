# SENPAI Research State — `ib-20260528-scen-c-r1`

- Date: 2026-05-28 (updated ~T+65)
- Active research tag: `ib-20260528-scen-c-r1`
- Advisor branch: `ib-20260528-scen-c-r1`
- Scope: Scenario C only (high-load, geomean throughput across burst/poisson/constant).
- Students: scen-c-frieren, scen-c-fern (1 GPU shared across both via `senpai/gpu_slot.py`).
- Hardware: 1× RTX PRO 6000 (shakedown mode).

## Most recent human research direction

- Operator launch: paper-parity Scenario C trial, ~2h budget, two logical students on one benchmark
  GPU. Require terminal results to come from supervised relaunch + `validate_result.py`.

## Live baseline

**22.18x** speedup_over_pytorch (Scenario C geomean). PR #163 merged. **SGLang 0.5.9**: triton attn + pytorch sampler, mem_fraction_static=0.88, max_running_requests=256, schedule_policy=fcfs, chunked_prefill_size=4096. W&B: b4xhsfby. Quality ratio 1.054 (n=500). VRAM peak 86.7 GB / 96 GB → 9.3 GB headroom remaining.

(Previous: vLLM Arm A at 20.84x, PR #161, W&B btpqa2rl — now rank 2.)

## Current research focus

- **Push SGLang concurrency further**: SGLang Arm A used `max_running_requests=256` and `mem_fraction_static=0.88` with 9.3 GB VRAM headroom unused. Next: push max_running_requests to 384/512 and mem_fraction_static to 0.90-0.92, possibly with chunked_prefill_size=8192. scen-c-fern's next PR.
- **Resolve chunked-prefill hypothesis**: frieren's PR #165 still WIP — full eval of vLLM Arm D (chunked-prefill OFF). If it beats 22.18x SGLang, both engines have leads. If it doesn't, vLLM Arm A stands as best-vLLM.
- **Hardcode winning SGLang config as launcher defaults** so reproduce-in-fresh-container doesn't depend on caller env vars. Build this into fern's next PR.

## Active PRs

- #165 (scen-c-frieren, vLLM Arm D + n-gram spec): WIP, full eval in progress.
- New fern PR (pending): SGLang concurrency push (max_running_requests 384/512, mem_fraction higher, chunked_prefill 8192) + hardcoded launcher defaults.

## Potential next research directions

- SGLang `--enable-torch-compile` (search space says false, but worth a quick probe at high concurrency).
- SGLang speculative decoding (`--speculative-algorithm EAGLE` or n-gram) for Scenario C's long-decode regime.
- FP8 weight quantization for SGLang (if booting + quality passes) — could push VRAM headroom even further.
- vLLM speculative decoding (n-gram) — Arm E in frieren's PR #165 if Arm D doesn't win.

## Stop rules / hard limits

- Reserve last 10-15 minutes for review + `BASELINE.md` update.
- No full evaluation may start unless `gpu_slot.py --mode full --min-remaining-s 1800` passes.
- Quality gate failures are not winners.
- All terminal updates must come through `validate_result.py`.
- Do NOT score Scenarios A, B, D in this launch.
