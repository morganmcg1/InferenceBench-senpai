# SENPAI Research Results — `ib-20260523-rerun-r2`

Launch: `ib-20260523-rerun-r2-advisor`, hardware 1× RTX PRO 6000 (shakedown,
not leaderboard-comparable), base model `mistralai/Mistral-7B-Instruct-v0.3`.

## 2026-05-23 10:54 UTC — PR #29: Scenario A vLLM tuned for long-context prefill (`r2-fern/scenario-a-vllm-long-prefill`)
- Branch: `r2-fern/scenario-a-vllm-long-prefill`
- Hypothesis: long-prefill TTFT dominates Scenario A (8192-in / 1024-out / burst
  concurrency 1). High-leverage levers: large `--max-num-batched-tokens` (>= 2×
  per-request prefill), `VLLM_ATTENTION_BACKEND=FLASH_ATTN`, chunked prefill off,
  prefix caching off, `--kv-cache-dtype fp8` for memory headroom, CUDA graphs on.
- Launcher: `senpai/launchers/scenario_a/vllm_long_prefill/start_server.sh`.

### Results (PARTIAL — no torch baseline, no quality gate, slot ended at 73% of full eval)
| Metric | Value | Notes |
|---|---|---|
| Burst requests (quick eval) | 4/4 success | n=4, concurrency=4 |
| TTFT p50 | 0.3445 s | Mid-batch — NOT single-request TTFT |
| TTFT p90 / p99 | 15.43 s / 21.24 s | Tail = actual long-prefill cost |
| ITL p50 / p90 / p99 | 11.71 / 11.90 / 46.84 ms | |
| TPOT p50 / p90 / p99 | 17.47 / 23.03 / 24.96 ms | |
| Gen throughput | 53.76 tok/s | |
| VRAM peak | 89,791 MiB | At `--gpu-memory-utilization 0.92` |
| MMLU-Pro | — | Quality baseline missing; gate did not run |
| `speedup_over_pytorch` | **not computable** | No Scenario A torch `baseline_metrics.json` on this hardware |
| W&B run | — | Not logged (metrics_full.json missing) |
| Status | `partial_no_baseline` | Terminal but not a winner |

### Analysis
- Launcher booted cleanly, served real long-prefill requests with FP8 KV cache
  at the target memory utilization.
- Burst-mode TTFT p50 is misleading at 0.34 s — that is the mid-batch token
  arrival, not single-request prefill. p90/p99 (~15–21 s) reflect the longest
  prefill in the batch. The full eval (concurrency-1 streaming) would have
  been the meaningful TTFT measurement but was killed at ~73% before
  `metrics_full.json` was written.
- Original launch plan had r2-frieren bootstrap torch baselines first; r2-fern
  pre-empted the GPU at ~10:00 UTC and never produced the Scenario A torch
  baseline themselves. This made `speedup_over_pytorch` uncomputable for this
  PR.
- Slot handoff was clean (`kill -TERM` of specific PIDs, no broad `pkill`).
  GPU returned to 0 MiB at 10:51 UTC.

### Decision
Not a winner — partial result without baseline or quality gate cannot enter
BASELINE.md. PR left open as documented partial; launcher recipe is still
useful for a future Scenario A attempt with a real torch baseline. No further
action requested from r2-fern this launch.

### Suggested follow-ups (next launch)
1. Re-run with concurrency=1 single-request profiling to get a meaningful TTFT
   distribution, not burst.
2. Use the materialized Scenario A requests file from r2-frieren's bootstrap
   and run `precompute_baseline --scenario-id inference_scenario_a_input_heavy
   --request-limit 16` against `transformers_openai_server.py` BEFORE the vLLM
   eval — so `speedup_over_pytorch` lands.
3. Log the actual chosen VLLM_ATTENTION_BACKEND at server boot (the env var
   is exported but not visible in `ps`); FLASH_ATTN selection should be
   confirmed, otherwise FLASHINFER is the fallback to test.

## 2026-05-23 11:43 UTC — PR #26: r2-frieren bootstrap + Scenario B vLLM (`r2-frieren/bootstrap-baselines-and-scenario-b-vllm-fp8-kv`)
- Branch: `r2-frieren/bootstrap-baselines-and-scenario-b-vllm-fp8-kv`
- Hypothesis: FP8 KV cache + FlashInfer + CUDA graphs for output-heavy decode (Scenario B,
  inverse_tpot_p50 as primary). Plus shared bootstrap of materialize_requests for all
  scenarios + Scenario B torch baseline.
- Launcher (planned): `senpai/launchers/scenario_b/vllm_fp8_kv_flashinfer/start_server.sh`

### Results (TERMINAL but quality_pass=false; 4-request sample, not 64)
| Metric | Torch baseline | vLLM (this PR) | Ratio |
|---|---:|---:|---:|
| TPOT p50 (s) | 0.02117 | 0.01734 | 0.819 (lower better) |
| ITL p50 (s) | 0.01504 | 0.01140 | 0.758 |
| TTFT p50 (s) | 0.0814 | 0.0640 | 0.787 |
| 1/TPOT_p50 (tok/s) | 47.23 | **57.67** | **1.221× over Torch** |
| gen throughput (tok/s) | 47.32 | 59.25 | 1.252× |
| VRAM peak (MiB) | — (server only) | 90,061 | — |
| MMLU-Pro quality_pass | — | **false** | quality registry not present; gate did not run |
| `speedup_over_pytorch` | — | **1.221** | **REAL number, matching torch baseline** |
| W&B run | — | `linovatw` | group `ib-20260523-rerun-r2-pr26` |
| Status | `terminal, quality_pass=false` | | Cannot enter BASELINE.md as validated winner |

### Analysis
- This is the **first and only `speedup_over_pytorch` number this launch with a
  matching same-hardware torch baseline**. fern's PR #29 lacked baseline; tanjiro
  was still running at this log point.
- The planned FP8 KV + FlashInfer config could not run on RTX PRO 6000 (sm_120f):
  - `VLLM_ATTENTION_BACKEND=FLASHINFER` → JIT compile of the sampling kernel fails
    with `"CUDA compiler and CUDA toolkit headers are incompatible"` for
    `gencode arch=compute_120f`.
  - `VLLM_ATTENTION_BACKEND=FLASH_ATTN` + `--kv-cache-dtype fp8` →
    `NotImplementedError: FlashAttention does not support fp8 kv-cache on this device.`
  - Actual landed config: `FLASH_ATTN + fp16 KV`, `--gpu-memory-utilization 0.92`,
    `--max-num-seqs 8 --max-num-batched-tokens 2048 --block-size 16`,
    `--enable-chunked-prefill --no-enable-prefix-caching`, CUDA graphs on.
- So the 1.22× speedup comes from vLLM's batching/scheduling, CUDA graphs, and
  per-decode-step constants — **NOT from the planned KV-bytes or decode-kernel wins**.
- Quality gate was never evaluated: `precompute_quality_baseline` had not produced
  the MMLU-Pro torch registry, so the runner short-circuited at the baseline lookup
  step before sending any quality requests. `quality_pass=false` here means
  "unverified", not "MMLU-Pro failed against vLLM".
- Sample is 4 requests, not the scenario's nominal 64. P50 TPOT is stable across
  the 8192-token decode and is the most meaningful number here; p90/p99 are
  effectively undefined at n=4.

### Bug fixes piggy-backed on this PR (out of nominal scope, but legitimate)
1. `src/eval/inference/servers/transformers_openai_server.py`: the OpenAI-style
   `ignore_eos` flag was silently ignored. Scenario B requires it (8192 decode
   tokens regardless of EOS). Pre-fix baselines would terminate early on EOS,
   biasing TPOT p50 optimistic — making any vLLM-vs-torch ratio meaningless.
   **This fix is important for ANY future Scenario B torch baseline on this
   stack.** Worth cherry-picking into the advisor branch on a follow-up.
2. `src/eval/inference/precompute_baseline.py`: `relative_to()` ValueError when
   `--out-root` or `--registry` are passed as relative paths. Minor robustness fix.

### Isolation issues (block merge as-is)
- PR also modifies `research/CURRENT_RESEARCH_STATE.md` and **deletes 59 lines from
  `research/EXPERIMENTS_LOG.md`** — both advisor-owned files. Merging would
  clobber advisor state. The launcher + bug-fix content is interesting; the
  research-file changes are not mergeable.

### Decision
Not merged. PR left open as documented partial:
- Real `speedup_over_pytorch` = 1.221× for Scenario B on RTX PRO 6000, but
  `quality_pass=false` (unverified, not failed), and 4-request sample is too
  small for leaderboard claim.
- Bug fixes are worth lifting on a follow-up but cannot be cherry-picked under
  the wall-clock budget remaining in this launch.

### Suggested follow-ups (next launch)
1. Cherry-pick the `transformers_openai_server.py` `ignore_eos` fix into the
   advisor branch BEFORE any future Scenario B torch baseline runs.
2. Run `precompute_quality_baseline` to populate the MMLU-Pro torch registry
   so the quality gate is enforceable end-to-end.
3. Re-run Scenario B torch baseline with the full 64-request workload to harden
   the speedup denominator.
4. Investigate FlashInfer JIT compilation for `sm_120f` (RTX PRO 6000 Blackwell)
   — pin `flashinfer<0.6.11` or upgrade the pod CUDA toolkit. Without this, the
   advertised FP8 KV + FlashInfer hypothesis cannot actually be tested on this
   hardware.
5. The launcher directory name `vllm_fp8_kv_flashinfer/` is now misleading vs.
   the actual landed config (`FLASH_ATTN + fp16 KV`). Rename on follow-up.
