# Live Advisor Baseline — ib-20260524-ready-r2

- **Updated:** 2026-05-24 (boot)
- **Mode:** RTX PRO 6000 shakedown (NOT leaderboard-comparable until repeated on H100)
- **Hardware:** 1x NVIDIA RTX PRO 6000 Blackwell, ~96 GB VRAM
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Time budget:** 2 hours total for the whole research program
- **W&B:** `wandb-applied-ai-team/inferencebench-senpai`
- **PyTorch baseline source:** imported scoring assets at `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248` (preflight green for A/B/C/D, seed 248)

## Starting launcher

`src/starting_points/vllm_running/start_server.sh` (vLLM default: `--gpu-memory-utilization 0.90 --max-model-len 131072`, no speculative decoding, no chunked prefill flag, default attention backend, eager off-default).

## Current best valid launcher (per scenario)

| Scenario | Primary metric | Speedup vs PyTorch | Launcher | PR | W&B run |
|---|---|---:|---|---|---|
| A: Input-heavy (TTFT p50) | `scenario/A/speedup_over_pytorch` | — | (none yet, starting point only) | — | — |
| B: Output-heavy (TPOT p50) | `scenario/B/speedup_over_pytorch` | **2.91x** | `senpai/launchers/B/ngram-spec-fp8-kv/start_server.sh` | #43 | `fgzox7fi` |
| C: High-load (req/s geomean) | `scenario/C/speedup_over_pytorch` | — | (none yet, starting point only) | — | — |
| D: General (balanced geomean) | `scenario/D/speedup_over_pytorch` | — | (none yet, starting point only) | — | — |

## Public reference (2026-05-21, H100, 2h budget) — for direction only, not a live target

| Method | Aggregate | A TTFT | B TPOT | C req/s | D geomean |
|---|---:|---:|---:|---:|---:|
| SMAC3 / vLLM 2h | 11.53x | 4.37x | 15.23x | 46.70x | 5.69x |
| vLLM default | 4.05x | 1.25x | 2.25x | 48.69x | 1.96x |
| PyTorch | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |

Headroom interpretation (RTX PRO 6000 is similar Blackwell-class, broadly comparable for direction): Scenario B has the largest gap (≈7x ceiling/default ratio for prefill+decode tuning + speculative decoding), then A (≈3.5x), then D (≈3x). Scenario C is already near-saturated at the vLLM default.

## Scenario B detail — current best

- **Launcher:** `senpai/launchers/B/ngram-spec-fp8-kv/start_server.sh` (PR #43, W&B `fgzox7fi`)
- **speedup_over_pytorch:** 2.913x (TPOT p50 8.64 ms vs PyTorch 25.15 ms; generation_throughput 98.28 tok/s)
- **Quality gate:** MMLU-Pro 0.3203 / baseline 0.298 / ratio 1.075 — PASS
- **Eval mode:** `--quick` n=8 burst requests (scenario-default n=64 would take ~89 min at 98 tok/s for 8192-out; deferred to next round)
- **Key config:** FLASH_ATTN backend (FlashInfer JIT fails on CUDA 13 pod), bf16 KV (FP8 KV wired via env var but inactive — same nvcc/header mismatch), n-gram spec `{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}`, max-num-seqs=16, CUDA graphs ON
- **Follow-up bets:** num_speculative_tokens=3 variant (+5–10%), EAGLE draft-model PR (unlocks 10x+ regime), full n=64 eval to confirm estimate

## Update history

- 2026-05-24 — File created, current best per-scenario unset, RTX PRO 6000 shakedown mode.
- 2026-05-24 07:58 — Scenario B best set to 2.91x (PR #43, r2-frieren, ngram-spec-fp8-kv). First winner merged.
