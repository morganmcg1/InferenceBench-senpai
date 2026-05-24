# BASELINE — ib-20260524-leasefix-r4

Live ledger maintained by the SENPAI advisor for this research tag.

## Setting

- **Hardware:** NVIDIA RTX PRO 6000 Blackwell (~96 GB VRAM) — shakedown mode, not leaderboard-comparable.
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **PyTorch baseline source:** `src/eval/inference/baselines/speed/torch/...`
  via `rtxpro6000-seed248` scoring-assets import.
- **Time budget:** 2 hour SENPAI window for the full research tag.
- **Starting launcher:** `src/starting_points/vllm_running/start_server.sh`
  (vLLM default — `--gpu-memory-utilization 0.90 --trust-remote-code --disable-log-stats`).
- **Runtime env:** students must `source senpai/runtime_env.sh` before launching
  (default disables implicit FlashInfer sampler/prefill, caps
  `INFERENCE_BENCH_MAX_MODEL_LEN=32768`).

## PyTorch baseline raw objectives (rtxpro6000-seed248)

| Scenario | Profile | TTFT p50 (s) | TPOT p50 (s) | Req/s | Raw objective | Pytorch raw |
|---|---|---:|---:|---:|---|---:|
| A | burst (conc 1) | 0.4385 | 0.0258 | 0.0709 | `1/ttft.p50` | 2.280 |
| B | burst (conc 1) | 0.0709 | 0.02515 | 0.0133 | `1/tpot.p50` | 39.76 |
| C | burst (conc 64) | 0.0702 | 0.02368 | 0.0845 | geomean req/s across 3 profiles | 0.0847 |
| C | poisson (32/s, cap 32) | 0.0702 | 0.02398 | 0.0847 | — | — |
| C | constant (16/s, cap 16) | 0.0704 | 0.02412 | 0.0849 | — | — |
| D | burst (conc 4) | 0.2123 | 0.02506 | 0.0382 | geomean(1/ttft.p50, 1/tpot.p50, req/s) | 1.184 |

All baselines: 100% success, no failures. Quality gate runs at `tau = 0.95`
against MMLU-Pro PyTorch baseline accuracy = 0.298 (mmlu_pro, seed=248,
n=500).

## Reference targets (H100 from program.md, not directly comparable here)

These are H100 SMAC3 search results from public InferenceBench leaderboard
(2026-05-21 snapshot). For RTX PRO 6000 they are informational, not goals:

| Scenario | SMAC3 (H100) | vLLM default (H100) |
|---|---:|---:|
| A | 4.37x | 1.25x |
| B | 15.23x | 2.25x |
| C | 46.70x | 48.69x |
| D | 5.69x | 1.96x |
| Aggregate (geomean A-D) | 11.53x | 4.05x |

## Current live baseline (this tag, RTX PRO 6000)

| Scenario | Best valid launcher | Best raw objective | Speedup over PyTorch | W&B run | PR |
|---|---|---:|---:|---|---|
| A | `senpai/launchers/scenario_a/vllm-fastprefill/start_server.sh` (vLLM 0.11.0 V1 + FLASH_ATTN + `--max-num-batched-tokens 16384`, `--max-num-seqs 8`, no prefix cache, CUDA graphs, `--gpu-memory-utilization 0.92`) | `1/ttft.p50 = 2.840` | **1.246x** | `7dew56hp` | #83 |
| B | (none yet) | — | — | — | — |
| C | `senpai/launchers/scenario_c/vllm-throughput/start_server.sh` (vLLM 0.11.0 V1 + FLASH_ATTN + `--max-num-seqs 256`, `--max-num-batched-tokens 8192`, `--enable-chunked-prefill`, no prefix cache, CUDA graphs, `--gpu-memory-utilization 0.92`) | geomean req/s = 0.3298 | **3.89x** (quick-only) | `8kxj9472` | #92 |
| D | (none yet) | — | — | — | — |

### Scenario A details (PR #83 — frieren)

- Full eval: 128/128 burst success, 0 failures.
- TTFT: p50 0.3521 s (PyTorch 0.4385 s, −19.7%); p90 0.3972 s; p99 0.4065 s.
- TPOT p50: 0.01708 s (PyTorch 0.02576 s, −33.7%).
- Request throughput: 0.0909 req/s (PyTorch 0.0709, +28.2%).
- Quality gate **PASS**: mmlu_pro observed = baseline = 0.298 (ratio 1.000, n=500).
- VRAM peak: 92 109 MB.
- Cold-relaunch confirmed.
- **Caveat (open scientific question):** vLLM 0.11.0 V1 engine silently
  force-enables chunked prefill regardless of `--no-enable-chunked-prefill`
  (`vllm/engine/arg_utils.py:1548`). The win is therefore not attributable to
  the no-chunked-prefill lever; it comes from FLASH_ATTN pinning, large
  `--max-num-batched-tokens`, CUDA graphs, and prefix-cache off.

### Scenario C details (PR #92 — frieren, QUICK-ONLY)

- **Quick eval only** (4 speed requests per profile, MMLU-Pro n=16).
  Round-2 full eval pending at n=256 + MMLU n=500.
- 12/12 speed requests successful, 0 failures, 0 empty outputs.
- Per-profile req/s: burst (conc 64) 0.3202, poisson (32/s cap 32) 0.3344,
  constant (16/s cap 16) 0.3349. Geomean = 0.3298 (PyTorch 0.0847 → **3.89x**).
- Per-profile speedup: burst 3.79x, poisson 3.95x, constant 3.95x.
- TPOT p50 ~0.020 s across all profiles; ITL p50 ~0.0117 s.
- Burst TTFT p50 0.650 s elevated by n=4 simultaneous prefill serialization
  against the 8 K chunked-prefill budget; expected to amortize at full n=256.
- VRAM peak: 90 173 MB. KV cache memory at boot: 72.87 GiB.
- Quality gate **FAIL** at n=16: observed 0.25 vs baseline 0.298, ratio 0.839,
  tau 0.95. Std err at n=16 ≈ 0.114; observed is 0.4σ below baseline,
  **consistent with noise** rather than a real regression. Round-2 full eval
  required to confirm.
- Cold-relaunch confirmed (supervised `launch_supervised_server.sh`).
- GPU slot wall-clock for full quick: ~75 s (acquired 23:12:55Z → released 23:14Z).

## Update history

- 2026-05-24: initialized live ledger from `rtxpro6000-seed248` PyTorch baselines.
- 2026-05-24 23:05Z: PR #83 merged. Scenario A: PyTorch → **1.246x** (frieren,
  W&B `7dew56hp`).
- 2026-05-24 23:18Z: PR #92 merged. Scenario C: PyTorch → **3.89x** quick-only
  (frieren, W&B `8kxj9472`). Quality gate noisy fail at n=16; round-2 full
  eval required.
