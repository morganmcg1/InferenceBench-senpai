# InferenceBench Live Baseline — `ib-20260524-ready-r1-advisor`

Live advisor-owned baseline ledger for the active SENPAI run. Compare every
terminal `SENPAI-RESULT` to the values in **Current Best Launcher** below.

## Run Context

- **Research tag:** `ib-20260524-ready-r1`
- **Advisor branch:** `ib-20260524-ready-r1-advisor`
- **Target base branch:** `codex/inferencebench-senpai-target`
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **GPU class (this run):** NVIDIA RTX PRO 6000 Blackwell (≈96 GB VRAM) —
  **shakedown evidence only**, not H100 leaderboard-comparable.
- **Time budget per training run:** `SENPAI_TIMEOUT_MINUTES` (per-run hard cap).
- **Whole-program clock:** 2 hours (assignment + smoke + full eval + review).
- **Seed / quality:** seed 248, MMLU-Pro tau=0.95.
- **Scoring assets:** `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`
- **Preflight (RTX PRO 6000):** PASS (all four scenarios, requests, baselines,
  and quality registry present; nvidia-smi WARN expected on advisor pod, tokenizer
  WARN expected without `INFERENCE_BENCH_ALLOW_HF_DOWNLOAD=1`).

## PyTorch Baseline (used to compute `speedup_over_pytorch`)

Use the matching baseline metrics file at:

```
src/eval/inference/baselines/speed/torch/<inference_scenario_*>/mistralai_Mistral-7B-Instruct-v0.3/baseline_metrics.json
```

after hydrating from the PVC import dir above. By definition the PyTorch
baseline `speedup_over_pytorch = 1.00x` for every scenario.

Key PyTorch baseline numbers on **RTX PRO 6000 seed 248** (from the JSONs above):

| Scenario | Profile | TTFT p50 (s) | TPOT p50 (s) | Gen tput (tok/s) | Req tput (r/s) |
|---|---|---:|---:|---:|---:|
| A burst | conc=1, 8K/1K  | 0.4385 | 0.02576 | 35.65 | 0.0709 |
| B burst | conc=1, 1K/8K  | 0.0709 | 0.02515 | 39.19 | 0.0133 |
| C burst | conc=64, 1K/1K | 0.0702 | 0.02368 | 39.19 | 0.0845 |
| C poiss | rate=32 r/s    | 0.0702 | 0.02398 | 38.39 | 0.0847 |
| C const | rate=16 r/s    | 0.0704 | 0.02412 | 38.62 | 0.0849 |
| D burst | conc=4, 4K/2K  | 0.2123 | 0.02506 | 38.21 | 0.0382 |

Primary metric per scenario (from `senpai/summarize_metrics.py`):

- **A:** burst-profile `1 / ttft_p50`; speedup = candidate(1/ttft) / baseline(1/ttft) = baseline_ttft / candidate_ttft.
- **B:** burst-profile `1 / tpot_p50`; speedup = baseline_tpot / candidate_tpot.
- **C:** geomean across burst/poisson/constant of `request_throughput_req_per_s`; speedup = geomean(candidate_rps) / geomean(baseline_rps).
- **D:** burst-profile geomean of three terms — `1/ttft_p50`, `1/tpot_p50`, `generation_throughput_tokens_per_s`; speedup = geomean(candidate triple) / geomean(baseline triple).

Quality gate: MMLU-Pro tau=0.95 against baseline accuracy ≈ 0.298 (seed 248, 500 samples).

## Starting Launcher

`src/starting_points/vllm_running/start_server.sh` — vLLM default with
`--gpu-memory-utilization 0.90`, no quantization, no speculative decoding,
no chunked prefill flag set explicitly. Treat as the unmeasured starting point
for the SENPAI run on RTX PRO 6000; replace as soon as a measured candidate
beats it on the target scenario.

## Current Best Launcher Per Scenario

| Scenario | Primary metric | Current best speedup | Recipe | W&B run | PR |
|---|---|---:|---|---|---|
| A: Input-heavy   | `scenario/A/speedup_over_pytorch` | — (no measured candidate yet) | starting vLLM default | — | — |
| B: Output-heavy  | `scenario/B/speedup_over_pytorch` | — (no measured candidate yet) | starting vLLM default | — | — |
| C: High-load     | `scenario/C/speedup_over_pytorch` | **46.35x** ✅ | `senpai/launchers/C/vllm-fp8-batch/start_server.sh` — vLLM 0.21 + FP8 weights+KV + chunked prefill + max-num-seqs 256 + max-num-batched-tokens 8192 + FLASH_ATTN + gpu-memory-utilization 0.92 | `yiumffcc` | #39 |
| D: General       | `scenario/D/speedup_over_pytorch` | — (no measured candidate yet) | starting vLLM default | — | — |

Public H100 reference snapshot (2026-05-21, **not on this hardware**, **do not
inherit**): SMAC3 2h vLLM gives A=4.37x, B=15.23x, C=46.70x, D=5.69x. SGLang
default gives C=51.12x. Use these only as direction signals; the live numbers
above are what every PR must beat.

## Update History

- 2026-05-24 07:54Z — PR #39 (r1-frieren, Scenario C) merged. First measured
  candidate on RTX PRO 6000 seed 248. **46.35x** speedup over PyTorch.
  Quality gate PASS: MMLU-Pro ratio 0.9530 (margin 0.0030 above 0.95 floor).
  Per-profile: burst 75.16x / poisson 47.84x / constant 27.69x.
  VRAM peak 90,775 MiB (94 %), cold relaunch healthy in ~38 s.
  W&B run: `yiumffcc`. Launcher: `senpai/launchers/C/vllm-fp8-batch/start_server.sh`.
- 2026-05-24 — ledger created. Preflight passed for RTX PRO 6000 seed 248.
  Three idle students; no measured candidates yet.
