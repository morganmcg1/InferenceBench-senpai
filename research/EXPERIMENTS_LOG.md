# SENPAI Research Results — InferenceBench ib-20260523-rerun-r3

## 2026-05-23 11:43 — PR #30: [r3-frieren] Sc. A FlashInfer + chunked prefill + FP8 KV + prefix caching for long-context TTFT

- **Branch:** `r3-frieren/scA-flashinfer-chunked-fp8kv`
- **Hypothesis:** Scenario A (8192-token prompts, concurrency 1) is bottlenecked by long-context prefill. FlashInfer attention backend (best Blackwell prefill kernel) + chunked prefill (max_num_batched_tokens=16384) + FP8 KV (frees activation budget) + prefix caching should cut TTFT meaningfully vs vLLM defaults.
- **Launcher:** `senpai/launchers/scenario_a/vllm-flashinfer-chunked-fp8kv/start_server.sh` (vLLM 0.21.0)
- **Required env vars:** `VLLM_USE_FLASHINFER_SAMPLER=0` (pod-environment workaround)

### Results

| Run | Reqs | TTFT.p50 (ms) | TTFT.p90 | 1/TTFT.p50 | gen_throughput | success | VRAM peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full eval (16 reqs, advisor-approved abbreviated) | 16/16 | **347** | 406 | **2.8825** | 58.78 tok/s | 100% | 91029 MiB |
| Cold relaunch quick (4 reqs) | 4/4 | 349 | 518 | 2.866 | 53.29 tok/s | 100% | 89879 MiB |

- **Primary metric (round-1 raw):** `scenario/A/raw/inverse_ttft_p50 = 2.8825`
- **Quality gate:** SKIPPED — `error="missing baseline accuracy for: mmlu_pro"` (pod-wide gap)
- **W&B run:** `dfgkjud9` (group `ib-r3-scA-flashinfer-chunked`, name `r3-frieren/vllm-flashinfer-chunked-fp8kv`)

### Analysis and conclusions

This is the **first measured Sc. A launcher** on this advisor branch and becomes the new Sc. A baseline.

1. **FlashInfer backend CONFIRMED engaged** via server.log line `Using AttentionBackendEnum.FLASHINFER backend.` — first such confirmation in this research session (r3-fern's Sc. B run used the dropped env var only, so its attention backend remains unconfirmed).

2. **TTFT.p50 = 347ms on 8192-token prefill** corresponds to ~23.6k tok/s prefill throughput on Blackwell — in the right neighborhood for FlashInfer's prefill kernel performance on a 96 GiB GPU with Mistral-7B BF16 weights.

3. **Cold relaunch in-distribution** (347 → 349ms), confirming the launcher is reproducibly fast across process restarts.

4. **Critical pod-environment workaround discovered:** `VLLM_USE_FLASHINFER_SAMPLER=0` must be set in any launcher that uses `--attention-backend FLASHINFER` on this pod, because the FlashInfer sampling JIT compile fails on missing `curand.h` (libcurand-13-2 runtime installed but no `-dev` headers package; `nvidia-curand-cu13` pip wheel is a 0.0.1 stub). The PyTorch-native sampler fallback is the right boundary — FlashInfer attention is preserved, only sampling switches.

5. **Lever attribution (informed by ablation logic):** FlashInfer is the only one of the four levers that directly accelerates the *first* token. FP8 KV does not help the first token (cache is written, not read) but frees activation budget for the larger `max-num-batched-tokens=16384`. Chunked prefill at 16384 ≥ 8192 keeps the whole prompt in one scheduler chunk. Prefix caching may show smaller wins on the burst's distinct prompts; spread p50→p99 (347→455ms) is narrow, suggesting prefix-cache hits aren't dominating.

6. **Quality gate not verifiable.** FP8 KV with e4m3 (no scaling factor configured) is the highest-risk lever for accuracy regression but cannot be measured without the missing MMLU-Pro baseline. Treat the TTFT result as conditional on quality.

### Next steps from r3-frieren's suggestions

- Restore quality baseline before round 2 (either materialize PyTorch quality baseline, OR adjust `quality_gate.py` to tolerate missing baseline with absolute threshold)
- A/B FP8 vs BF16 KV cache to isolate any TTFT cost of FP8 writes
- Sweep `max_num_batched_tokens` ∈ {8192, 16384} for chunk-size sensitivity
- Try `--enable-chunked-prefill=False` at concurrency=1 (chunking may add scheduler overhead unnecessarily)
- Tune `--cudagraph-capture-sizes 1` (concurrency=1 only uses bs=1 graph, capturing full ladder wastes warmup time)
- Add `--baseline-metrics-json` to log_metrics_to_wandb to compute speedup once a PyTorch baseline exists
- Pod-level fix: install cuda-toolkit-13-2 / curand dev headers so FlashInfer sampler can JIT

## 2026-05-23 11:25 — PR #32: [r3-fern] Sc. B n-gram speculative decoding + FP8 KV + FlashInfer for long-decode TPOT

- **Branch:** `r3-fern/scB-ngram-spec-fp8kv`
- **Hypothesis:** Scenario B (output-heavy, concurrency 1, 8192-token decode) is bottlenecked by per-token decode latency. N-gram speculative decoding (5-token drafts from local prompt-lookup buffer) combined with FP8 KV cache and CUDA graphs can achieve large TPOT speedup.
- **Launcher:** `senpai/launchers/scenario_b/vllm-ngram-spec-fp8kv/start_server.sh` (vLLM 0.21.0, `--max-num-seqs 8 --max-num-batched-tokens 4096 --kv-cache-dtype fp8 --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}' --gpu-memory-utilization 0.92`)

### Results

| Run | Reqs | tpot.p50 (ms) | 1/tpot.p50 (tok/s) | gen_throughput | ttft.p50 | success | vram peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| Quick eval (pre-relaunch) | 4/4 | 6.027 | 165.93 | 167.63 tok/s | 49.6 ms | 100% | 90069 MB |
| Partial full eval (8 reqs) | 8/8 | 7.895 | **126.67** | 133.42 tok/s | 33.0 ms | 100% | 90123 MB |
| Quick eval (cold relaunch) | 4/4 | 4.966 | 201.37 | 191.53 tok/s | 49.0 ms | 100% | 90069 MB |

- **Primary metric (round-1 raw):** `scenario/B/inverse_tpot_p50 = 126.67 tok/s` (8-req partial full eval)
- **Quality gate:** SKIPPED — quality baseline registry absent on this pod
- **W&B run:** `qixqiu9x` (group `ib-r3-scB-ngram-spec`, name `r3-fern/vllm-ngram-spec-fp8kv`)
- **Speculative config confirmed:** `method='ngram', num_spec_tokens=5`, `kv_cache_dtype=fp8`, `cudagraph_mode=FULL_AND_PIECEWISE`, `enforce_eager=False`

### Analysis and conclusions

This is the **first measured Sc. B launcher** on this advisor branch and becomes the new Sc. B baseline.

1. **N-gram speculative decoding engaged correctly** on vLLM 0.21.0. CUDA graph capture at batch sizes [1,2,4,...,96] and full PIECEWISE mode confirms the low-latency decode path is active.

2. **126.67 tok/s on 8 reqs vs H100 vLLM-default reference (~37 tok/s derived)** — preliminary directional signal that the speculative + FP8 KV recipe transfers to Blackwell. Without the per-pod PyTorch baseline this is not leaderboard-comparable.

3. **Variance is high at small N** (quick=166 → partial=127 → cold-quick=201 tok/s) due to KV/compile cache warm state and sample-mix effects. A full 64-req eval would tighten this significantly.

4. **FlashInfer backend status unknown** for this run: launcher used the dropped `VLLM_ATTENTION_BACKEND` env var (vLLM 0.21 API change discovered by r3-frieren). Actual backend likely FlashAttention2. Future runs should use `--attention-backend FLASHINFER` CLI flag.

5. **Two tokenizer bugs patched in `senpai/materialize_requests.py`** (now merged to advisor branch):
   - `_count_chat_tokens` returned BatchEncoding(len=2) for Mistral's chat template; fixed with robust `len(ids)` using `.input_ids`
   - `_truncate_messages` decode→re-encode drift caused `realized < min_input_tokens`; fixed with `robust_truncate_messages` scanning ±8 token budgets

### Next steps from r3-fern's suggestions

- Run full 64-req Sc. B + PyTorch baseline to produce leaderboard `speedup_over_pytorch`
- Sweep `num_speculative_tokens ∈ {3, 5, 7}` vs acceptance rate
- Add `--attention-backend FLASHINFER` and re-test to isolate FlashInfer vs FlashAttn2 contribution
- Stage quality baseline registry on the pod for future quality gate runs
