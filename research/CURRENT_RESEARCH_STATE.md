# SENPAI Research State — InferenceBench

- **Last updated:** 2026-05-23 11:48 UTC (round 1 closed — zero leaderboard data, tooling carryover only)
- **Most recent research direction from human researcher team:** (none yet for
  this launch — boot only)
- **Research tag:** `ib-20260523-rerun-r1`
- **Advisor branch:** `ib-20260523-rerun-r1-advisor`
- **Hardware (shakedown):** 1 x NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM
- **Wall-clock budget (round 1):** 2 hours total — consumed

## Current research focus and themes

The research goal is to discover inference-serving configurations and systems
recipes for `mistralai/Mistral-7B-Instruct-v0.3` that beat the default vLLM
launcher's PyTorch-relative speedup on each scenario, while preserving the
MMLU-Pro quality gate and integrity rules. All four scenarios remain open for
improvement:

- **A: Input-heavy (TTFT p50, burst conc 1)** — prefill latency on 8192-token
  prompts. Levers: chunked prefill size, max-num-batched-tokens, attention
  backend (FlashInfer/Flash/Triton), CUDA graphs, prefix caching off (concurrency
  1 so likely zero hit), eager-vs-compiled.
- **B: Output-heavy (TPOT p50, burst conc 1)** — long decode on 8192-token
  generations. Levers: speculative decoding (n-gram), FP8 KV cache, chunked
  prefill, kv-cache dtype, attention backend, gpu-memory-utilization.
- **C: High-load (req/s geomean across burst, poisson, constant)** — throughput
  on 256 concurrent requests. Levers: max-num-seqs, max-num-batched-tokens,
  prefix caching, kv-cache fp8, scheduler policy, GPU memory utilization.
- **D: General (geomean TTFT/TPOT/throughput, burst conc 4)** — a balanced
  serving profile. Reusable launchers from A/B/C are good candidates.

## Round 1 outcome (closed 2026-05-23 11:48 UTC)

Zero leaderboard data across all three PRs. Three compounding blockers consumed
the 2-hour wall budget:

1. **vLLM 0.11.0 → 0.21.0 upgrade** (~14 min) — container shipped vLLM 0.11.0
   built for CUDA 12 / torch 2.8.0, but pod has CUDA 13.2 / torch 2.11.0. ABI
   mismatch in `vllm._C`. r1-tanjiro landed the upgrade with a long manual dep
   chain and unblocked the fleet.
2. **Tokenizer roundtrip off-by-one** in `senpai/materialize_requests.py`
   blocked pre-materialization on scenarios B, C, D (seed=248 lands realized
   tokens 1 below the random target's min bound). r1-frieren wrote a
   `robust_truncate_messages` iterative converger patch (commit `999179b`).
3. **FlashInfer JIT needs `curand.h`** which is missing from the pod's CUDA 13
   toolkit. Both r1-tanjiro's startup attempts failed at JIT compile — first
   the sampler, then the attention kernel (even with
   `VLLM_USE_FLASHINFER_SAMPLER=0`).

A fourth latent constraint: torch backend is slow (~70 sec/request). Sc A
torch baseline alone (128 requests) projects 75-90 min, which would consume
the entire post-tooling wall budget. r1-frieren's first precompute run died
across a Claude session boundary (no detach), and the relaunched run was
killed at 11:28 UTC per the option-R pivot decision to give r1-tanjiro a
no-denominator vLLM measurement window before harvest start.

All three PRs are closed:
- **PR #20** (r1-frieren Sc C) — closed, tooling carryover cherry-picked to
  advisor branch.
- **PR #22** (r1-fern Sc B) — closed, launcher recipe preserved on branch.
- **PR #23** (r1-tanjiro Sc A) — closed, launcher + toolchain diagnostic
  preserved on branch.

Tooling carryover now on advisor branch (cherry-picked commits `999179b` and
`52ef8387`):
- `senpai/materialize_requests.py` with `robust_truncate_messages` iterative
  converger.
- Materialized `requests.jsonl` for scenarios A, B, D under both
  `src/eval/inference/baselines/speed/default/<scenario>/` and
  `src/eval/inference/baselines/speed/torch/<scenario>/<safe_model>/`.
  (Sc C requests are NOT materialized — round 2 must run the patched
  materialize_requests on Sc C to produce them.)
- MMLU-Pro samples cache at
  `src/eval/inference/baselines/samples/mmlu_pro/248_500/samples.jsonl`
  (seed=248, n=500, ~500KB).
- Sc C `aggro-batch-fp8-kv` launcher recipe preserved at
  `senpai/launchers/C/aggro-batch-fp8-kv/start_server.sh`.

Pod-level issue logged for the platform team:
- CUDA 13 toolkit needs `nvidia-curand-cu13` (or `curand.h` + `curand_kernel.h`
  copied into `/usr/local/cuda/include/`) at pod-build time so FlashInfer's
  JIT compile path works. This is a hard blocker for any FlashInfer-based
  hypothesis on this pod.

## Round 2 priorities (in order)

Strategy: round 2 must produce the **torch baselines + MMLU-Pro quality
registry** that round 1 never landed, then run a small set of low-risk
non-FlashInfer candidate launchers. FlashInfer is benched until the pod
toolchain is fixed.

**Slot 1 (highest priority): torch baselines + quality registry.**
- Run `precompute_all_baselines.py --cache-quality-samples --backends torch
  --scenarios <subset>` with `nohup setsid` detach (so it survives Claude
  session boundaries).
- Materialized request files for A, B, D already exist (round-1 carryover),
  so the precompute should skip the slow request-sampling step for those.
- Sc C requests still need materialization with the patched
  `senpai/materialize_requests.py`.
- Pick a scenario subset that fits the wall budget. Sc A torch is
  ~75-90 min for 128 requests; Sc B torch is comparable or longer because
  decode time dominates; Sc C torch is ~30-60 min (smaller request counts in
  poisson/constant profiles); Sc D torch is comparable to A.
- **Highest-leverage option:** build Sc A + Sc D baselines first
  (their requests count is 128 and 96 respectively, both are burst-conc-1
  prefill-dominant which torch handles fastest), plus the quality registry.
  That's ~150-180 min. Need an explicit longer round-2 wall budget OR build
  in parallel by serializing per-scenario quality+speed runs.

**Slot 2: non-FlashInfer Sc A `attn-fallback` candidate.**
- New launcher variant of `r1-tanjiro/scenario-a-flashinfer-fp8` that uses
  `--attention-backend FLASH_ATTN` (or no `--attention-backend` flag, letting
  vLLM pick the default for CUDA 13) but keeps `--kv-cache-dtype fp8` +
  `--max-num-batched-tokens 16384` + `--block-size 16` + the existing
  `--max-model-len 32768` fix.
- Tests the FP8-KV + big-prefill-batch hypothesis without the FlashInfer
  JIT gap.
- Should boot. Confirms whether the speedup comes from the attention backend
  or the batching + FP8 KV.

**Slot 3: Sc B n-gram speculative-decoding candidate.**
- Re-run r1-fern's `r1-fern/scenario-b-ngram-spec` launcher (already verified
  to dry-parse cleanly on vLLM 0.21) against the new Sc B torch baseline.
- Watch the quality gate carefully — n-gram speculative decoding can degrade
  MMLU-Pro accuracy. If quality gate fails, fall back to the same launcher
  without `--speculative-config`.

**Slot 4 (if time): Sc C aggressive batching candidate.**
- r1-frieren's `senpai/launchers/C/aggro-batch-fp8-kv/start_server.sh` (now
  on the advisor branch) is the candidate. Sc C torch baseline must exist
  first; if slot-1 prioritizes A+D baselines, Sc C is a round-3 candidate.

**Tooling improvements queued (advisor or whichever student has spare time):**
- Fix `senpai/summarize_metrics.py::_baseline_primary_value` path mismatch
  flagged by r1-fern at 10:52 UTC (calls `primary_metric(baseline_metrics,
  scenario)` directly on baseline JSON wrapped under `{"baseline": {...}}`).
- Pod-build patch for `curand.h` — coordinate with the platform team if
  available, otherwise document as a known limitation and route all
  FlashInfer hypotheses around it.

## Round-2 hypothesis catalog

`research/RESEARCH_IDEAS_2026-05-23_round2.md` holds 12 ideas across all 4
scenarios:
- EAGLE-3 spec decode
- SGLang RadixAttention
- TGI tuned
- TensorRT-LLM
- AWQ / FP8-weights quantization
- Attention backend ablations
- Cross-scenario confirmation runs
- (Plus carryover from round 1: r1-frieren's Sc C launcher, r1-tanjiro's Sc A
  launcher with `attn-fallback` variant, r1-fern's Sc B launcher)

Round 2 priorities above selected from this catalog with the **bootability
on this pod** filter applied — FlashInfer-based recipes are deprioritized
until the curand.h gap is fixed.

## Potential next research directions (longer horizon)

- **SGLang as alternative engine** on whichever scenario shows the weakest vLLM
  result (especially Scenario A/D where SGLang's RadixAttention can help, but
  burst concurrency 1 may erase that).
- **TGI** as a fallback engine for stability.
- **Speculative decoding variants:** longer `num_speculative_tokens` for
  Scenario B; EAGLE / Medusa if vLLM supports a path from CLI.
- **Quantization sweep:** GPTQ / AWQ / FP8 weights — only if quality gate still
  passes.
- **Attention backend ablation:** FLASH_ATTN vs FLASHINFER vs TRITON_ATTN per
  scenario (FlashInfer pending pod toolchain fix).
- **Scheduler policy ablation:** `--scheduling-policy` priority vs fcfs for
  Scenario C poisson/constant traffic.
- **Cross-scenario confirmation runs** for mature winners — once a launcher
  beats baseline on one scenario, sanity-check A-D for the geomean leaderboard.
- **Compile/graph capture caveats** on RTX PRO 6000 Blackwell — measure
  warm-up cost before declaring CUDA graph wins.
