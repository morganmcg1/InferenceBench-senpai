# SENPAI Research Results — InferenceBench ib-20260523-rerun-r3

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
