# InferenceBench SENPAI Live Baseline — ib-20260524-hardened-r5

Living advisor ledger. Compare every terminal review-ready PR to this state and
update when a candidate becomes the new current best for a scenario.

- **Setting**: RTX PRO 6000 Blackwell, 96GB VRAM, single GPU, 2 hour SENPAI window
- **Model**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Scoring assets (PVC)**: `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`
- **Dataset/quality seed**: 248
- **MMLU-Pro gate**: tau = 0.95 × baseline accuracy 0.298 = **observed ≥ 0.283**
- **Status of leaderboard claims**: Shakedown evidence only; not leaderboard-comparable until repeated on H100.

## PyTorch Reference (this hardware)

These are the per-scenario PyTorch baseline raw numbers measured on this GPU.
Speedup = candidate_raw / PyTorch_raw.

| Scenario | Profile | PyTorch ttft.p50 | PyTorch tpot.p50 | PyTorch req/s | Raw objective used |
|---|---|---:|---:|---:|---|
| A: input-heavy (8192/1024, burst c=1) | burst | 0.4385 s | 0.02576 s | 0.0709 | `1/ttft.p50(burst)` ≈ 2.280 |
| B: output-heavy (1024/8192, burst c=1) | burst | 0.0709 s | 0.02515 s | 0.0133 | `1/tpot.p50(burst)` ≈ 39.76 |
| C: high-load (1024/1024 ×3 profiles) | burst | 0.0702 s | 0.02368 s | 0.0845 | geomean(req/s) ≈ 0.0847 |
|   | poisson | 0.0702 s | 0.02398 s | 0.0847 |   |
|   | constant | 0.0704 s | 0.02412 s | 0.0849 |   |
| D: general (4096/2048, burst c=4) | burst | 0.2123 s | 0.02506 s | 0.0382 | geomean(1/ttft, 1/tpot, req/s) burst |

## Public H100 Reference (2026-05-21, leaderboard ladder)

| Method | Aggregate | A TTFT | B TPOT | C req/s | D geomean |
|---|---:|---:|---:|---:|---:|
| SMAC3 search 2h vLLM | 11.53× | 4.37× | 15.23× | 46.70× | 5.69× |
| TPE search 2h vLLM | 11.25× | 4.48× | 14.76× | 43.46× | 5.58× |
| vLLM default | 4.05× | 1.25× | 2.25× | 48.69× | 1.96× |
| PyTorch | 1.00× | 1.00× | 1.00× | 1.00× | 1.00× |

These H100 ratios are context only. We score against the PyTorch raw numbers in
the row above on RTX PRO 6000.

## Current best valid launcher per scenario

| Scenario | Current best speedup | PR | W&B run | Launcher path | Notes |
|---|---:|---|---|---|---|
| A | **1.889×** | #63 | f890wobr, h2ivavpf | `senpai/launchers/A/vllm-fp8-flashinfer-prefill-scA/start_server.sh` | FP8 weights + auto KV + FLASH_ATTN + chunked-prefill OFF; FlashInfer+FP8-KV broken on SM120f |
| B | 1.00× | — | — | — | starting point: `src/starting_points/vllm_running/start_server.sh` |
| C | 1.00× | — | — | — | starting point: `src/starting_points/vllm_running/start_server.sh` |
| D | 1.00× | — | — | — | starting point: `src/starting_points/vllm_running/start_server.sh` |

## Known RTX PRO 6000 runtime constraints (SM120f, CUDA 13.2, vLLM 0.11)

- **FlashInfer JIT broken**: `CPATH` set by `senpai/runtime_env.sh` mixes CUDA 12.8 pip headers with CUDA 13.2 nvcc; FlashInfer JIT fails. `runtime_env.sh` now exports `VLLM_USE_FLASHINFER_SAMPLER=0` and `VLLM_DISABLE_FLASHINFER_PREFILL=1`. Avoid `VLLM_ATTENTION_BACKEND=FLASHINFER` unless explicitly testing.
- **FP8 KV cache broken**: `FLASH_ATTN` raises `NotImplementedError: FlashAttention does not support fp8 kv-cache on this device` on SM120f. Do not use `--kv-cache-dtype fp8` with the FLASH_ATTN backend.
- **Max model len**: `INFERENCE_BENCH_MAX_MODEL_LEN` defaults to 32768 in `senpai/runtime_env.sh`; Mistral config caps at 32768. Use 16384 or 32768, not 131072.
- **FP8 weights**: Safe and boots cleanly; not affected by the above constraints. Use `--quantization fp8` freely.

## Update history

- 2026-05-24 15:55 — initialized live baseline for advisor branch `ib-20260524-hardened-r5`.
- 2026-05-24 17:45 — merged PR #63 (r5-fern): Scenario A 1.889× speedup (TTFT.p50 0.232s vs baseline 0.439s, MMLU-Pro 0.294, clean relaunch 0.13% drift). FP8 weights + FLASH_ATTN + chunked-prefill-OFF on RTX PRO 6000.
