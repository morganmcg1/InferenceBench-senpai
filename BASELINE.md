# InferenceBench SENPAI Baseline Ledger

Live ledger for advisor branch `ib-20260524-hardened-r2`. Compares per-scenario
results against the PyTorch baseline (raw) and tracks the best launcher
recipe known on this branch. Update on every terminal review-ready PR.

- **Mode:** RTX PRO 6000 shakedown (NOT leaderboard-comparable)
- **Time budget:** 2 hours total program wall-clock
- **Base model:** mistralai/Mistral-7B-Instruct-v0.3
- **Hardware:** 1× NVIDIA RTX PRO 6000 Blackwell (~96 GB VRAM), shared by 3 students
- **Quality gate:** MMLU-Pro 500 questions @ seed=248, tau=0.95 × PyTorch baseline accuracy
- **PyTorch baseline accuracy:** 0.298 (MMLU-Pro)
- **Preflight assets:** `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248` (PASS)
- **W&B project:** `wandb-applied-ai-team/inferencebench-senpai`

## PyTorch Baselines (measured on RTX PRO 6000, seed=248)

| Scenario | Profile | Raw objective | Value |
|---|---|---|---:|
| A | burst | 1 / ttft.p50 | 2.281 (ttft.p50 = 0.4385 s) |
| B | burst | 1 / tpot.p50 | 39.67 (tpot.p50 = 0.0252 s) |
| C | geomean(burst, poisson, constant) | request_throughput_req_per_s | 0.0847 |
| D | burst | geomean(1/ttft, 1/tpot, rps) | ~4.86 (ttft=0.2123, tpot=0.0251, rps=0.0382) |

## RTX PRO 6000 Compatibility Constraints (live discoveries)

Discovered during ib-20260524-hardened-r2 round 1. All new launcher recipes must respect:
- **NO `VLLM_ATTENTION_BACKEND=FLASHINFER`** — FlashInfer JIT fails on sm_120 (Blackwell) due to CUDA 12.8 wheel vs nvcc 13.2 CCCL version mismatch.
- **NO `--kv-cache-dtype fp8` with FlashAttention** — FlashAttention rejects fp8 KV on this device.
- **`--kv-cache-dtype fp8` works with TRITON_ATTN** — r2-tanjiro confirmed this on PR #64.
- **Use `VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_DISABLE_FLASHINFER_PREFILL=1`** — set by updated `runtime_env.sh` to prevent vLLM from entering FlashInfer paths automatically.
- **`INFERENCE_BENCH_MAX_MODEL_LEN=32768`** — Mistral-7B max_position_embeddings is 32768; vLLM 0.11 rejects 131072 without explicit override.
- **Always `source "$PROBLEM_DIR/senpai/runtime_env.sh"`** before launching a server.

## Current Best Launcher (this branch)

| Scenario | Best speedup_over_pytorch | PR | Launcher | W&B run | Notes |
|---|---:|---|---|---|---|
| A | unmeasured | – | – | – | r2-fern eval in progress (PR #58) |
| B | unmeasured | – | – | – | r2-frieren eval pending (PR #54) |
| C | unmeasured | – | – | – | not yet assigned |
| D | **1.284×** | #64 (merged) | `senpai/launchers/D/balanced-fp8kv-chunked/start_server.sh` | m4w2qrxd | TRITON_ATTN + FP8 KV + chunked prefill; quality ratio=0.953 |

### Scenario D — Best: 1.284× (PR #64, merged 2026-05-24)

- **Launcher:** `senpai/launchers/D/balanced-fp8kv-chunked/start_server.sh`
- **vLLM flags:** `--kv-cache-dtype fp8 --enable-chunked-prefill --max-num-batched-tokens 8192 --max-num-seqs 64 --gpu-memory-utilization 0.92`
- **Attention backend:** TRITON_ATTN (via `VLLM_USE_FLASHINFER_SAMPLER=0`, default backend, no `VLLM_ATTENTION_BACKEND` override)
- **Speedup:** 1.284× over PyTorch (raw geomean 2.478 vs PyTorch 1.930)
- **Key metrics (burst, concurrency=4):** ttft.p50=0.192s (−9.5%), tpot.p50=0.0165s (−34%), rps=0.0483 (+26%), gen_tps=57.1 (+49%)
- **Quality:** MMLU-Pro 0.284/0.298, ratio=0.953 (PASS, margin tight — avoid further FP8 quantization)
- **VRAM:** 90.9 GB / 97.9 GB
- **W&B:** `wandb-applied-ai-team/inferencebench-senpai/runs/m4w2qrxd`
- **Student suggested follow-ups:** prefix caching, max-num-batched-tokens sweep (4096/16384), max-num-seqs sweep

## Public Reference (H100, 2h budget, for direction only — NOT comparable on RTX PRO 6000)

| Method | Aggregate | A | B | C | D |
|---|---:|---:|---:|---:|---:|
| SMAC3 search 2h vLLM | 11.53x | 4.37x | 15.23x | 46.70x | 5.69x |
| TPE search 2h vLLM | 11.25x | 4.48x | 14.76x | 43.46x | 5.58x |
| vLLM default | 4.05x | 1.25x | 2.25x | 48.69x | 1.96x |
| PyTorch | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |

## Update History

- 2026-05-24 09:XX — initialized advisor ledger; preflight passed via imported PVC assets.
- 2026-05-24 17:30 — merged PR #64 (r2-tanjiro, Sc D); Sc D baseline now 1.284×; confirmed TRITON_ATTN + FP8 KV is the bootable recipe on RTX PRO 6000; FlashInfer JIT confirmed broken on sm_120.
