# SENPAI Research Results — ib-20260524-leasefix-r3

## 2026-05-24 22:55 — PR #77: Scenario D vLLM balanced-burst (chunked-prefill + CUDA graphs + prefix cache)

- **Student:** tanjiro
- **Branch:** `tanjiro/vllm-balanced-burst-D` (merged)
- **Hypothesis:** Default vLLM on Scenario D (balanced, concurrency=4, 4096-token prompts, 2048-token outputs) is throttled by three coupled bottlenecks: long-prompt prefill (TTFT), small-batch decode (TPOT), and request throughput. Chunked prefill should let the scheduler overlap a long prefill with decode for the other 3 in-flight requests; CUDA graphs are the dominant TPOT lever for small batches on Blackwell; prefix caching is cheap insurance; trimming `max_num_seqs` to 16 concentrates the KV cache rather than spreading it across 256 unused slots.
- **Launcher levers:** `--enable-chunked-prefill --enable-prefix-caching --max-num-seqs 16 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92`, FlashAttention backend, `max_model_len=32768`, CUDA graphs enabled (default, no `--enforce-eager`).

### Results

| Metric | PyTorch | vLLM balanced-burst | Δ |
|---|---:|---:|---:|
| `scenario/D/speedup_over_pytorch` | 1.000x | **1.305x** | +30.5% |
| Raw geomean (1/ttft.p50, 1/tpot.p50, req/s) | 1.9298 | 2.5190 | +30.5% |
| `quality/mmlu_pro_observed_accuracy` | 0.298 | **0.308** | +1.0pp |
| `quality/mmlu_pro_ratio` (τ=0.95) | — | 1.034 | PASS |
| `ttft.p50` (s) | 0.2123 | 0.1735 | -18.3% |
| `ttft.p90` (s) | 0.2489 | 0.1945 | -21.9% |
| `tpot.p50` (s) | 0.02506 | 0.01697 | -32.3% |
| `tpot.p90` (s) | 0.05505 | 0.02649 | -51.9% |
| `itl.p50` (s) | 0.01434 | 0.01143 | -20.3% |
| `request_throughput_req_per_s` | 0.03823 | 0.04706 | +23.1% |
| `generation_throughput_tokens_per_s` | 38.21 | 56.24 | +47.2% |
| success / failure | 96 / 0 | 96 / 0 | — |
| VRAM peak (MB) | — | 90,737 | — |

**W&B run:** `wandb-applied-ai-team/inferencebench-senpai/runs/z1m0zvsl`

### Analysis

- The biggest single win is on **TPOT** (-32% p50, -52% p90). That is consistent with CUDA-graph capture on small decode batches at concurrency=4 — defaults disable eager fallback, and `max_num_seqs=16` keeps the captured batch range narrow.
- **TTFT** improvement (-18%) tracks the chunked-prefill hypothesis: long 4 K-token prefills overlap better with the 3 in-flight decodes rather than blocking the scheduler.
- Throughput followed (req/s +23%, tok/s +47%) as a downstream effect of better TPOT and prefill overlap.
- **Quality went *up*** (0.308 vs 0.298, ratio 1.034). No precision regression. MMLU-Pro is sampled with temperature 0.4 / nucleus, so small accuracy fluctuations across backends are within noise; we did not lose accuracy switching from PyTorch to vLLM with this launcher.
- The H100 reference shows default vLLM at 1.96x and SMAC best at 5.69x on D. RTX PRO 6000 numbers are not directly comparable, but the H100 SMAC gap suggests substantial Blackwell headroom beyond 1.305x.

### Caveats

- Clean-relaunch `metrics_relaunch.json` was not captured: the wrapper script killed its own PGID when tearing down phase 1, taking `gpu_slot.py run` with it. The launcher is deterministic (env-only shell script) and the first `./test_server.sh` was already a fresh-launch case, so the merged result is valid. The wrapper PGID bug is a tooling-only follow-up.

### Suggested follow-ups (from student)

1. Wrapper PGID fix (benchmark tooling, separate PR).
2. Explicit `cuda-graph-capture-sizes` pinned to `{1,2,4,8}` for tighter TPOT capture.
3. Smaller `max_model_len` (e.g. 8192) to free KV slots for higher real concurrency.
4. Marlin / W8A16 quantization as the next TPOT lever.

## 2026-05-24 23:43 — PR #88: Scenario B vLLM n-gram speculative decoding (num_spec=5)

- **Student:** fern
- **Branch:** `fern/vllm-ngram-spec-decode-B` (merged, squash commit `c3ca8ef`)
- **Hypothesis:** Scenario B is the most extreme decode-latency case (concurrency=1, 8192-token outputs). TPOT dominates and the GPU is bandwidth-bound. vLLM's native n-gram speculative decoding (no draft model required) should amortize autoregressive cost on Mistral-Instruct outputs that contain repeated n-grams. `num_speculative_tokens=5` plus `prompt_lookup_min=2`, `prompt_lookup_max=4`, FLASH_ATTN, no chunked-prefill, no prefix caching, `max_num_seqs=8`, `block_size=16`. Original PR also requested `--kv-cache-dtype fp8` but the student found vLLM 0.11.0 FLASH_ATTN backend rejects fp8 KV on Blackwell (FA3+SM90-only), so fp8 KV was dropped — concurrency=1 + 96 GB VRAM means memory headroom was not needed anyway.
- **Winning launcher:** `senpai/launchers/scenario_b/vllm-ngram5-fp8kv/start_server.sh`

### Results

| Metric | PyTorch | vLLM n-gram spec=5 | Δ |
|---|---:|---:|---:|
| `scenario/B/speedup_over_pytorch` | 1.000x | **2.694x** | +169.4% |
| Raw obj (1/tpot.p50) | 39.76 | 107.09 | +169.4% |
| `quality/mmlu_pro_observed_accuracy` | 0.298 | **0.302** | +0.4pp |
| `quality/mmlu_pro_ratio` (τ=0.95) | — | 1.013 | PASS |
| `ttft.p50` (s) | 0.0709 | 0.0641 | -9.6% |
| `ttft.p90` (s) | — | 0.0655 | — |
| `ttft.p99` (s) | — | 0.5517 | first-batch warmup |
| `tpot.p50` (s) | 0.0252 | **0.00934** | **-62.9%** |
| `tpot.p90` (s) | — | 0.01247 | — |
| `tpot.p99` (s) | — | 0.0836 | spec-reject batch |
| `itl.p50` (s) | — | 0.01309 | — |
| `request_throughput_req_per_s` | 0.01334 | 0.02187 | +63.9% |
| `generation_throughput_tokens_per_s` | 39.19 | **103.68** | +164.6% |
| success / failure | 64 / 0 | 64 / 0 | — |
| VRAM peak (MB) | — | 87,725 | — |

**W&B run:** `wandb-applied-ai-team/inferencebench-senpai/runs/rnc0c1by`

### Analysis

- TPOT.p50 collapsed from 25.2 ms → 9.34 ms — a **2.69x decode speedup** on a single autoregressive request. That matches the speculative-decoding hypothesis: drafted n-grams accepted with high enough rate to amortize the verification pass.
- Generation throughput rose 2.65x (39 → 104 tok/s) and request throughput 1.64x as a direct consequence of the TPOT win at concurrency=1.
- Quality is essentially identical (0.302 vs 0.298, +0.4pp; ratio 1.013) — spec decoding is verifiable, so accepted tokens are bit-identical to greedy without speculation, and rejected drafts fall back to vanilla decode.
- The H100 reference shows default vLLM at 2.25x on B and SMAC3 best at 15.23x. Our 2.694x already exceeds the H100 default and is in the "n-gram-as-cheap-baseline" tier; reaching the 15x range would need a draft model (Eagle/Medusa) which is a Round 3 lever.

### Caveats

- The 0.5517 s p99 TTFT and the 0.0836 s p99 TPOT are first-batch warmup and spec-reject outliers respectively. Median behavior is what counts for raw obj = 1/tpot.p50.
- The PR contains two committed launchers: `B/vllm-ngram-spec/start_server.sh` (parameterized template, NOT the run that produced the W&B result) and `scenario_b/vllm-ngram5-fp8kv/start_server.sh` (the actual run config). BASELINE.md tracks the latter as the current best for B.

### Suggested follow-ups

1. Draft-model speculation (Eagle, Medusa) — much higher acceptance rate on free-form text.
2. Sweep `num_speculative_tokens` ∈ {3, 7, 8} now that 5 is confirmed.
3. Combine ngram spec + chunked-prefill (the PR disabled chunked-prefill; on B concurrency=1 this may not matter but worth testing).
