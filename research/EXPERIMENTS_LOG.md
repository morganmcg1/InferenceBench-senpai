# SENPAI Research Results — InferenceBench `ib-20260523-rerun-r1`

Append one entry per reviewed PR. Most recent at the bottom of each PR
section is fine; new PR sections appear in PR-number order or chronological
order, whichever is easier.

## Round 1 (2026-05-23, ~09:48-11:48 UTC, 1xRTX PRO 6000 Blackwell, ~96 GB VRAM)

**Outcome: ZERO leaderboard data across all three students.** Three blockers compounded across the 2-hour wall budget:

1. The container shipped vLLM 0.11.0 (built for CUDA 12.x + torch 2.8.0) but the pod has CUDA 13.2 + torch 2.11.0 — vLLM `_C` ABI mismatch. r1-tanjiro upgraded to vLLM 0.21.0 with a long manual dep install (~14 min into the budget) and unblocked the fleet.
2. The senpai `materialize_requests.py` tokenizer roundtrip lost 1 token at boundary, causing `realized_input_tokens >= min_input_tokens` to fail on Sc B/C/D pre-materialization. r1-frieren wrote a `robust_truncate_messages` iterative converger patch (commit `999179b`).
3. r1-tanjiro's FlashInfer-attention launcher hit a curand.h missing-header gap in the pod's CUDA 13 toolkit — FlashInfer JIT cannot compile its sampling or attention kernels.

The PyTorch torch backend was also slow (~70 sec/request) so the Sc A torch baseline (128 requests) projected ~75-90 min, which alone consumed the entire post-tooling wall budget. Advisor pivoted to option-R at 11:21 UTC: kill all baselines, let r1-tanjiro run a no-denominator vLLM candidate. That candidate hit the FlashInfer JIT gap on both startup attempts.

### 2026-05-23 11:43:11 — PR #20: r1-frieren — Scenario C: aggressive vLLM batching (chunked-prefill + FP8 KV + bigbatch)

- Branch: `r1-frieren/scenario-c-aggro-batch`
- Head commits: `999179b senpai: patch _truncate_messages to converge to target tokens`, `52ef8387 senpai: commit round-1 carryover (launcher + materialized requests A/B/D)`
- Hypothesis (planned): Sc C high-load throughput from `--max-num-seqs 512 --max-num-batched-tokens 16384 --enable-chunked-prefill --enable-prefix-caching --kv-cache-dtype fp8 --gpu-memory-utilization 0.95 --block-size 16` on vLLM 0.21.0.
- Terminal SENPAI-RESULT @ 11:31:56 UTC: `{"terminal":true,"status":"blocked-on-baseline","pending_arms":false,"wandb_run_ids":[],"primary_metric":{"name":"scenario/C/speedup_over_pytorch","value":null},"test_metric":{"name":"quality/mmlu_pro_observed_accuracy","value":null}}`

| Round-1 deliverable | Result |
|---|---|
| Sc C launcher executed | ❌ (never ran — slot repurposed to baselines, then cut) |
| `scenario/C/speedup_over_pytorch` | null (no measurement) |
| `quality/mmlu_pro_observed_accuracy` | null (no measurement) |
| Torch baseline produced | ❌ (precompute reached 30/128 on Sc A by 10:49, died at session boundary; relaunched 11:13:57 with `nohup setsid`; killed at 11:28 per option-R pivot — reached 54/128) |
| MMLU-Pro quality registry built | ❌ |
| MMLU-Pro samples cached | ✅ (seed=248, n=500, ~500KB, at `src/eval/inference/baselines/samples/mmlu_pro/248_500/samples.jsonl`) |
| Materialized `requests.jsonl` for Sc A, B, D | ✅ (under both `speed/default/<scenario>/` and `speed/torch/<scenario>/<safe_model>/`) |
| `robust_truncate_messages` patch | ✅ (commit `999179b` in `senpai/materialize_requests.py`) |
| Sc C launcher recipe preserved | ✅ (at `senpai/launchers/C/aggro-batch-fp8-kv/start_server.sh`) |

**Action taken:** Closed. Both commits cherry-picked to advisor branch so round 2 inherits them. Sc C `requests.jsonl` is NOT in the cherry-pick (Sc C still has the truncation bug; round-2 Sc C student must run materialize_requests with the new patch).

**Process notes:** Advisor closed this PR prematurely at 10:49:54 UTC (16 sec after the student's 10:49:38 status update) due to a race condition between the close-comment composition and the student's late-but-legitimate progress post. PR was reopened at 11:11:26 UTC and the slot order reverted to original — but the recovery cost ~22 min of round-1 budget. New advisor pre-close re-read protocol is in effect for future rounds.

### 2026-05-23 11:43:11 — PR #22: r1-fern — Scenario B: n-gram speculative decoding + FP8 KV

- Branch: `r1-fern/scenario-b-ngram-spec`
- Head commit: `e83e230 Scenario B: stage n-gram spec + FP8 KV launcher and research notes`
- Hypothesis (planned): Sc B output-heavy decode acceleration via vLLM n-gram speculative decoding (window 5, max_ngram 3, num_speculative_tokens 5) + FP8 KV cache.
- Terminal SENPAI-RESULT @ 11:12:24 UTC: `{"terminal":false,"status":"blocked-on-missing-baseline","pending_arms":false,"wandb_run_ids":[],"primary_metric":{"name":"scenario/B/speedup_over_pytorch","value":null},"test_metric":{"name":"quality/mmlu_pro_observed_accuracy","value":null}}`

| Round-1 deliverable | Result |
|---|---|
| Sc B launcher executed | ❌ (never ran — no Sc B torch baseline; r1-fern stood down at advisor direction at 11:11 UTC) |
| Launcher recipe preserved | ✅ (at `senpai/launchers/B/ngram-spec-fp8-kv/start_server.sh`) |
| Dry-parse verification on vLLM 0.21 | ✅ (JSON `--speculative-config` form + `--kv-cache-dtype fp8` both accepted) |
| `/tmp/ib-B/` workspace stage script | ✅ |
| `summarize_metrics.py::_baseline_primary_value` bug flagged | ✅ (path mismatch: function calls `primary_metric(baseline_metrics, scenario)` directly on baseline JSON which is wrapped under `{"baseline": {...}}` — would always raise `missing burst profile`) |

**Action taken:** Closed. Launcher preserved on the branch for round-2 re-assignment. Launch-isolation discipline noted (r1-fern explicitly declined to borrow r1-frieren's `robust_truncate_messages` patch).

### 2026-05-23 11:43:11 — PR #23: r1-tanjiro — Scenario A: FlashInfer attention + FP8 KV + big prefill batch

- Branch: `r1-tanjiro/scenario-a-flashinfer-fp8`
- Head commits: `f3c452d Add Scenario A FlashInfer+FP8 KV+big prefill batch launcher`, `0568b2e Fix max-model-len default for Mistral-7B-Instruct-v0.3`, `1cb2a0c Disable FlashInfer sampler in Scenario A launcher`
- Hypothesis (planned): Sc A burst-conc-1 TTFT speedup via vLLM 0.21.0 with `--attention-backend FLASHINFER --max-num-batched-tokens 16384 --enable-chunked-prefill --kv-cache-dtype fp8 --block-size 16 --gpu-memory-utilization 0.95 --max-num-seqs 8 --max-model-len 32768`.
- Terminal SENPAI-RESULT @ 11:43:07 UTC: `{"terminal":true,"status":"blocked-on-toolchain","pending_arms":false,"wandb_run_ids":[],"primary_metric":{"name":"scenario/A/raw_objective_inverse_ttft_p50","value":null},"test_metric":{"name":"quality/mmlu_pro_observed_accuracy","value":null}}`

| Round-1 deliverable | Result |
|---|---|
| Sc A launcher executed | ❌ (server failed to boot on both attempts) |
| `scenario/A/raw_objective_inverse_ttft_p50` | null (no measurement) |
| Launcher recipe preserved | ✅ (at `senpai/launchers/A/flashinfer-fp8-bigbatch/start_server.sh`, head `1cb2a0c`) |
| Mistral-7B `--max-model-len 32768` fix | ✅ (commit `0568b2e`) |
| `VLLM_USE_FLASHINFER_SAMPLER=0` workaround | ✅ (commit `1cb2a0c`) |
| vLLM 0.21.0 dependency landing for fleet | ✅ (gguf, prometheus_client, uvloop, watchfiles, setproctitle, xgrammar, prometheus-fastapi-instrumentator, cbor2, pyzmq, blake3, partial-json-parser, msgspec, lm-format-enforcer, outlines-core, diskcache, lark, py-cpuinfo, openai-harmony, openai>=2.0.0, mistral_common, compressed-tensors, depyf, python-json-logger, transformers>=4.56.0, tokenizers>=0.21.1, pybase64, ijson, fastapi[standard], aiohttp>=3.13.3, tiktoken, sentencepiece, llguidance, ninja, cloudpickle, model-hosting-container-standards, mcp, opentelemetry-{sdk,api,exporter-otlp,semantic-conventions-ai}, anthropic — manual install needed because container shipped vLLM 0.11.0 against CUDA 12 / torch 2.8.0) |
| FlashInfer-on-CUDA-13 toolchain blocker documented | ✅ (curand.h missing; full reproduction in PR #23 11:43:07 comment) |

**Attempt 1** (11:31:47 UTC): full PR-instructed flags. Failed at FlashInfer sampler JIT: `flashinfer/sampling.cuh:20:10: fatal error: curand.h: No such file or directory`.

**Attempt 2** (11:37:50 UTC): same flags + `VLLM_USE_FLASHINFER_SAMPLER=0`. Failed at FlashInfer attention JIT during cudagraph warmup, same curand.h missing.

**Action taken:** Closed. The launcher is sound on paper (all flags parse, model loads in 4.6s using 13.51 GiB) but unbootable on this pod until cuRAND headers land. Round-2 follow-ups (priority order): (1) install `nvidia-curand-cu13` or copy `curand.h`/`curand_kernel.h` into `/usr/local/cuda/include/` at pod-build time; (2) `-attn-fallback` variant with `--attention-backend FLASH_ATTN` keeping FP8-KV + big-prefill-batch; (3) pre-warmed FlashInfer JIT cache.

### Round-1 advisor self-review

**What worked:**
- All 3 students respected GPU-CLAIM/SLOT-FREE coordination after the first slot (zero compute-contention incidents).
- r1-tanjiro's pre-emptive vLLM 0.21 upgrade (with later advisor ratification) unblocked the fleet 30+ min faster than a serialized handoff would have.
- r1-fern's launch-isolation discipline (refusing to borrow r1-frieren's `robust_truncate_messages` patch) preserved provenance and propagation went through advisor cherry-pick.
- Cluster-cutoff verification (reading `/mnt/new-pvc/senpai-start-gates/...`) was correct and load-bearing at 11:20 UTC pivot decision.

**What broke:**
- The advisor's premature close on PR #20 at 10:49:54 (16-second race with the student's 10:49:38 status update) caused ~22 min of round-1 budget loss to recovery. New advisor protocol: `pr_all_comments` re-check immediately before any cut action.
- Round 1 launched without pre-built PyTorch baselines and without a pre-built MMLU-Pro quality registry. With a 2h wall budget and ~70 sec/request on torch, building the baselines consumes the budget — the round needs them pre-staged.
- The pod's CUDA 13 toolkit gap (missing `curand.h`) is a pod-build issue, not a student issue. It would have killed any FlashInfer recipe in round 1.

**Bug-fixes / tooling carryover cherry-picked to advisor branch:**
- `999179b` — `senpai/materialize_requests.py` `robust_truncate_messages` patch.
- `52ef8387` — materialized `requests.jsonl` for Sc A, B, D under `speed/default/` and `speed/torch/<safe_model>/`; MMLU-Pro samples cache (seed=248, n=500); Sc C `aggro-batch-fp8-kv` launcher recipe.
