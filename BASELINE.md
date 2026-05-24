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

Until a student PR establishes a measured vLLM-default-on-this-hardware
result, the current live baseline is the PyTorch baseline above.

| Scenario | Best valid launcher | Best raw objective | Speedup over PyTorch | W&B run | PR |
|---|---|---:|---:|---|---|
| A | (none yet) | — | — | — | — |
| B | (none yet) | — | — | — | — |
| C | (none yet) | — | — | — | — |
| D | (none yet) | — | — | — | — |

## Update history

- 2026-05-24: initialized live ledger from `rtxpro6000-seed248` PyTorch baselines.
