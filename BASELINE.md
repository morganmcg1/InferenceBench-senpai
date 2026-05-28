# SENPAI InferenceBench Baseline Ledger

- **Advisor branch:** `ib-20260528-12h-r2`
- **Research tag:** `ib-20260528-12h-r2`
- **Hardware (current):** NVIDIA RTX PRO 6000 (~96 GB VRAM) — shakedown only,
  not leaderboard-comparable to H100 reference snapshot.
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **W&B:** `wandb-applied-ai-team/inferencebench-senpai`
- **Scoring assets:** imported from
  `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`
  (seed 248, all four scenarios). Preflight: PASS as of 2026-05-28 10:42 UTC.

Reference snapshot for the H100 80GB / 2-hour-per-run leaderboard setting
(2026-05-21, copied from `program.md` for orientation only — not RTX PRO 6000
baselines):

| Method | Sc. A TTFT | Sc. B TPOT | Sc. C req/s | Sc. D geomean |
|---|---:|---:|---:|---:|
| PyTorch baseline | 1.00x | 1.00x | 1.00x | 1.00x |
| vLLM default | 1.25x | 2.25x | 48.69x | 1.96x |
| SGLang default | 1.22x | 1.77x | 51.12x | 2.14x |
| HF TGI default | 1.14x | 1.37x | 41.94x | 1.80x |
| Random search vLLM 2h | 4.21x | 11.34x | 41.81x | 5.42x |
| TPE search vLLM 2h | 4.48x | 14.76x | 43.46x | 5.58x |
| SMAC3 search vLLM 2h | 4.37x | 15.23x | 46.70x | 5.69x |
| Sonnet 4.6 agent | 3.47x | 12.03x | 33.93x | 3.01x |

## Current best valid launcher (RTX PRO 6000 shakedown)

| Scenario | Primary metric | Best speedup vs PyTorch | Launcher | W&B run | PR |
|---|---|---:|---|---|---|
| **A** | scenario/A/speedup_over_pytorch | **1.866x** | `senpai/launchers/A/frieren-vllm-ttft/arm3_fp8_weights.sh` | izg22lch | #137 |
| **B** | scenario/B/speedup_over_pytorch | **2.687x** | `senpai/launchers/B/fern-vllm-tpot/arm3_ngram_spec.sh` | dav3txgq | #136 |
| C | scenario/C/speedup_over_pytorch | 1.00x (PyTorch floor) | — | — | — |
| D | scenario/D/speedup_over_pytorch | 1.00x (PyTorch floor) | — | — | — |

### Scenario B — current winner (PR #136, merged 2026-05-28)

- **Engine:** vLLM 0.11.0, FlashAttention backend, n-gram (prompt-lookup) speculative decoding, BF16 KV cache
- **Key flags:** `--speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}' --max-num-seqs 16 --max-num-batched-tokens 2048 --enable-chunked-prefill --no-enable-prefix-caching --gpu-memory-utilization 0.92`
- **TPOT.p50:** 0.00936 s (PyTorch 0.0252 s) — exact verification, zero quality risk from speculation
- **Speedup:** 2.687x (inverse_tpot_p50: 106.84 vs 39.76 tok/s)
- **Quality:** MMLU-Pro 0.300 obs / 0.298 baseline = ratio 1.007 (gate 0.95, n=500) ✓
- **Speed success:** 64/64 (failure_rate 0.0) ✓
- **VRAM peak:** 90305 MiB / 97887 MiB
- **Reproduce (from task workspace):** `cp senpai/launchers/B/fern-vllm-tpot/arm3_ngram_spec.sh ./start_server.sh && python evaluate.py --json-output-file metrics_full.json`

### Scenario A — current winner (PR #137, merged 2026-05-28)

- **Engine:** vLLM 0.11.0, FlashAttention backend, FP8 weight-only quantization, BF16 KV cache
- **Key flags:** `--quantization fp8 --max-num-seqs 8 --max-num-batched-tokens 10240 --no-enable-chunked-prefill --no-enable-prefix-caching --gpu-memory-utilization 0.92`
- **TTFT.p50:** 0.2349 s (PyTorch 0.4385 s)
- **Speedup:** 1.866x (inverse_ttft_p50: 4.256 vs 2.280)
- **Quality:** MMLU-Pro 0.288 obs / 0.298 baseline = ratio 0.966 (gate 0.95, n=500) ✓
- **Speed success:** 128/128 (failure_rate 0.0) ✓
- **VRAM peak:** 93267 MiB / 97887 MiB
- **Reproduce (from task workspace):** `cp senpai/launchers/A/frieren-vllm-ttft/arm3_fp8_weights.sh ./start_server.sh && python evaluate.py --json-output-file metrics_full.json`

PyTorch baseline raw objectives (from imported metrics):

| Scenario | Profile | TTFT.p50 | TPOT.p50 | Throughput (req/s) |
|---|---|---:|---:|---:|
| A | burst | 0.4385s | 0.0258s | 0.0709 |
| B | burst | 0.0709s | 0.0252s | 0.0133 |
| C | burst | 0.0702s | 0.0237s | 0.0845 |
| C | poisson | 0.0702s | 0.0240s | 0.0847 |
| C | constant | 0.0704s | 0.0241s | 0.0849 |
| D | burst | 0.2123s | 0.0251s | 0.0382 |

## Provisional / quick-only results

These are `--quick` runs (`request_limit=4`, `quality_n=16`). Quality ratios at
this sample size are statistical noise, not gate-eligible. Speedups are
research signal only and must not be used to update the current-best row.

| Scenario | Arm | Quick speedup | Quality ratio (n=16) | W&B run | PR | Notes |
|---|---|---:|---:|---|---:|---|
| B | fern arm1 BF16 tuned | 1.438x | 0.839 | ex8p5t6j | #136 | vLLM 0.11, max_num_seqs=16, chunked prefill on |
| B | fern arm2 FP8 weights | 1.474x | 0.839 | ycwnvt87 | #136 | +`--quantization fp8`; within 5% of arm1 |
| B | fern arm3 n-gram spec | **3.518x** | 0.839 | du9y0jz8 | #136 | +n-gram spec, num_speculative_tokens=5 — promoted to full eval |
| A | frieren arm1 one-shot prefill | 1.276x | 0.839 | 5iump62l | #137 | vLLM 0.11, no chunked prefill flag (vLLM forces it anyway) |
| A | frieren arm2 chunked prefill large | 1.279x | 0.839 | r20z2rm0 | #137 | within noise of arm1 |
| A | frieren arm3 FP8 weights | **1.901x** | 0.839 | eyvyj8oz | #137 | +`--quantization fp8` — promoted to full eval |
| D | tanjiro arm1 sglang_default | 1.167x | 0.629 | 2nfds9ud | #138 | SGLang per-PR venv; **launcher not relaunch-safe (libnuma host install)** |
| D | tanjiro arm2 sglang_tuned | 1.183x | 0.629 | 8zul6lqz | #138 | +chunked-prefill-size=4096, lpm scheduler; within 5% of arm1 |

## Update history

- 2026-05-28 10:42 UTC — Created the baseline ledger. Preflight passed for all
  scenarios; assigning first round of experiments to fern, frieren, tanjiro.
- 2026-05-28 11:45 UTC — Logged round-1 quick-probe research signal. All three
  students promoted their best arm to full eval. No terminal results yet.
  Tanjiro PR #138 sent back for libnuma packaging / vLLM-fallback decision
  before any SGLang full eval can become terminal.
- 2026-05-28 11:53 UTC — Merged PR #137 (frieren). Scenario A new best:
  1.866x (FP8 weight quantization, vLLM 0.11, FlashAttention). Quality 0.966,
  128/128 speed success. First terminal win on this branch.
- 2026-05-28 12:19 UTC — Merged PR #136 (fern). Scenario B new best:
  2.687x (n-gram speculative decoding, num_speculative_tokens=5, prompt_lookup_max=4,
  vLLM 0.11, FlashAttention). Quality ratio 1.007 (observed 0.300, n=500),
  64/64 speed success. Beats H100 vLLM default (2.25x) and SGLang default (1.77x).
