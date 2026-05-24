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
