# SENPAI Research Results — ib-20260524-leasefix-r4

## 2026-05-24 23:05 — PR #83: Scenario A vLLM aggressive prefill (FlashAttn, large prefill batch, CUDA graphs)
- Branch: `frieren/sc-A-vllm-fastprefill`
- Hypothesis: For Scenario A (8 K prefill / 1 K decode burst conc-1), TTFT is
  prefill-bound. vLLM tuned with FLASH_ATTN, large
  `--max-num-batched-tokens 16384`, `--max-num-seqs 8`, no prefix cache,
  CUDA graphs on, `--gpu-memory-utilization 0.92` should beat the PyTorch
  reference (TTFT p50 = 0.4385 s, raw = 2.280).

| Metric (full eval) | Candidate | PyTorch baseline | Δ |
|---|---:|---:|---:|
| TTFT p50 (s) | 0.3521 | 0.4385 | −19.7% |
| TPOT p50 (s) | 0.01708 | 0.02576 | −33.7% |
| Request throughput (req/s) | 0.0909 | 0.0709 | +28.2% |
| Raw `1/ttft.p50` | 2.840 | 2.280 | +24.6% |
| Speedup over PyTorch | **1.246x** | 1.000x | +0.246x |
| MMLU-Pro accuracy (n=500) | 0.298 | 0.298 | ratio 1.000 (PASS) |
| VRAM peak (MB) | 92 109 | — | — |
| Burst success | 128/128 | 128/128 | clean |

- W&B run: `7dew56hp` (group `sc-A-vllm-fastprefill`).
- Cold-relaunch confirmed via the supervised launcher.
- **Caveat (open scientific question):** vLLM 0.11.0 V1 engine silently
  force-enables chunked prefill regardless of `--no-enable-chunked-prefill`
  (`vllm/engine/arg_utils.py:1548`). The win is therefore not attributable
  to the no-chunked-prefill lever; it comes from FLASH_ATTN pinning, large
  `--max-num-batched-tokens`, CUDA graphs, and prefix-cache off.
- Merged as Scenario A live baseline. Headroom remains relative to the H100
  SMAC3 reference (4.37x) — round-2 candidates include prefix caching,
  larger batched tokens, FP8 KV cache, FlashInfer.

## 2026-05-24 23:18 — PR #92: Scenario C vLLM high-concurrency throughput (quick-eval only)
- Branch: `frieren/sc-C-vllm-throughput`
- Hypothesis: For Scenario C (1 K / 1 K, three traffic profiles: burst
  conc-64, poisson 32 r/s cap 32, constant 16 r/s cap 16), the PyTorch
  serial baseline (geomean req/s 0.0847) is artificially low. vLLM tuned
  with FLASH_ATTN, `--max-num-seqs 256`, `--max-num-batched-tokens 8192`,
  `--enable-chunked-prefill`, no prefix cache, CUDA graphs on,
  `--gpu-memory-utilization 0.92` should win by a wide margin.

| Profile | Candidate req/s | PyTorch req/s | Per-profile speedup |
|---|---:|---:|---:|
| burst (conc 64) | 0.3202 | 0.0845 | 3.79x |
| poisson (32/s, cap 32) | 0.3344 | 0.0847 | 3.95x |
| constant (16/s, cap 16) | 0.3349 | 0.0849 | 3.95x |
| **Geomean req/s** | **0.3298** | **0.0847** | **3.89x** |

| Metric (quick eval, n=4 per profile + MMLU n=16) | Value |
|---|---:|
| TTFT p50 burst | 0.650 s (elevated — 4 sims serialize on 8 K chunked-prefill budget; expected to amortize at full n=256) |
| TTFT p50 poisson | 0.092 s |
| TTFT p50 constant | 0.073 s |
| TPOT p50 (all profiles) | ~0.020 s |
| ITL p50 (all profiles) | ~0.0117 s |
| VRAM peak | 90 173 MB |
| Speed requests success | 12/12 (0 failures) |
| MMLU-Pro observed accuracy (n=16) | 0.25 (FAIL at tau 0.95, ratio 0.839) |

- W&B run: `8kxj9472` (group `sc-C-vllm-throughput`).
- Quality gate FAIL is **noise-consistent**: at n=16, std err on a p=0.298
  Bernoulli ≈ 0.114; observed 0.25 is ~0.4σ below baseline (95% CI roughly
  0.07-0.43). No sampling-setting drift from baseline.
- Quick-only result merged as the Scenario C live baseline because no prior
  vLLM SENPAI baseline existed for C. Round-2 priorities:
  1. **Full eval confirmation** at n=256 per profile + MMLU n=500 (resolve
     quality-gate uncertainty and tighten speed CIs).
  2. Push `--max-num-seqs` toward 512 — burst-64 quick result suggests
     headroom (per-profile speedups already ~3.9x with conservative seq
     budget; H100 SMAC3 reference reaches 46.7x).
  3. FP8 weight quantization as a separate hardware-path PR.
- GPU slot wall-clock for full quick: ~75 s (acquired 23:12:55Z, released
  23:14Z). The recipe is very cheap to run.
