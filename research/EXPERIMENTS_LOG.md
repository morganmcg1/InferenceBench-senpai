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
