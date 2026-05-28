# SENPAI Research State — `ib-20260528-scen-c-r1`

- Date: 2026-05-28 (updated ~T+40)
- Active research tag: `ib-20260528-scen-c-r1`
- Advisor branch: `ib-20260528-scen-c-r1`
- Scope: Scenario C only (high-load, geomean throughput across burst/poisson/constant).
- Students: scen-c-frieren, scen-c-fern (1 GPU shared across both via `senpai/gpu_slot.py`).
- Hardware: 1× RTX PRO 6000 (shakedown mode).

## Most recent human research direction

- Operator launch: paper-parity Scenario C trial, ~2h budget, two logical students on one benchmark
  GPU. Require terminal results to come from supervised relaunch + `validate_result.py`.

## Live baseline

**20.84x** speedup_over_pytorch (Scenario C geomean). PR #161 merged. vLLM, Arm A: max_num_seqs=384, max_num_batched_tokens=16384, gpu_mem_util=0.92, chunked_prefill ON, BF16 KV. W&B: btpqa2rl.

## Current research focus

- **Resolve chunked-prefill hypothesis**: Quick probe in PR #161 showed Arm D (prefill OFF) +37% over Arm A in quick mode, but student predicted this is a small-N artifact that reverses in full eval. scen-c-frieren's next PR tests this via full eval of Arm D directly (no new quick needed).
- **Validate SGLang on RTX PRO 6000**: scen-c-fern (PR #163) successfully booted SGLang 0.5.9 with triton backend and passed quality at A0. Arm B (max_running_requests=256, lpm policy, chunked-prefill=4096, mem-fraction=0.88) is the next probe. SGLang reference on H100 is 51.12x — it may significantly outperform the current vLLM baseline.
- **N-gram speculative decoding**: Scenario C has 1024-token outputs with ignore_eos=true — high-value target for n-gram speculation. Assigned as Arm E to frieren as a fallback if Arm D doesn't beat the baseline.

## Active PRs

- #163 (scen-c-fern, SGLang): Arm A0 done (3.88x quick, quality pass). Arm B next. Status: WIP.
- New frieren PR (pending): Arm D full eval → resolve chunked-prefill question. Then Arm E (n-gram spec decode) if time.

## Potential next research directions

- If SGLang beats vLLM baseline: explore `--schedule-policy lpm` vs `fcfs` more aggressively.
- FP8 weight quantization + BF16 KV (not FP8 KV) — valid path on this hardware.
- Larger `--max-num-batched-tokens` (32768) if full eval shows batched tokens are the bottleneck.
- Continuous batching tuning for SGLang's RadixAttention.

## Stop rules / hard limits

- Reserve last 10-15 minutes for review + `BASELINE.md` update.
- No full evaluation may start unless `gpu_slot.py --mode full --min-remaining-s 1800` passes.
- Quality gate failures are not winners.
- All terminal updates must come through `validate_result.py`.
- Do NOT score Scenarios A, B, D in this launch.
