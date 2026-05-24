# InferenceBench Live Baseline — ib-20260524-hardened-r4

This is the advisor-owned live baseline ledger for SENPAI research tag
`ib-20260524-hardened-r4`. Each terminal review-ready PR is ranked against the
current best column. The PyTorch numbers come from the imported PVC scoring
assets (`/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`),
generated 2026-05-24 with `senpai/require_scoring_preflight.sh --import-dir ...`.

## Run setting

- Hardware (shakedown): 1× NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM
- Leaderboard claim hardware: 1× NVIDIA H100 80 GB (not yet exercised in this tag)
- Base model: `mistralai/Mistral-7B-Instruct-v0.3`
- Seed: 248
- Time budget: 2 hours per SENPAI run window
- W&B project: `wandb-applied-ai-team/inferencebench-senpai`
- Quality gate: MMLU-Pro 500 questions, tau=0.95 of PyTorch baseline accuracy
  0.298. Required ratio ≥ 0.95.
- Preflight: passed 2026-05-24 (hardware confirmed in PVC manifest as
  `NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB`; advisor pod has no
  GPU so the live nvidia-smi check warned, baselines/requests/quality all pass).

## PyTorch reference per scenario (RTX PRO 6000, seed 248)

| Scenario | Primary metric direction | Profile(s) | Raw objective at 1.00x | Notes |
|---|---|---|---|---|
| A: Input-heavy (8192 in / 1024 out, 128 reqs, burst conc 1) | maximize `1/ttft.p50` | burst | `ttft.p50 = 0.4385 s` → obj=2.281 | 100% success |
| B: Output-heavy (1024 in / 8192 out, 64 reqs, burst conc 1) | maximize `1/tpot.p50` | burst | `tpot.p50 = 0.02515 s` → obj=39.76 | 100% success |
| C: High-load (1024 in / 1024 out, 256 reqs each) | maximize geomean req/s | burst c=64, poisson 32 rps cap 32, constant 16 rps cap 16 | geomean `req/s ≈ 0.0847` | 100% success across all 3 profiles |
| D: General (4096 in / 2048 out, 96 reqs, burst conc 4) | maximize geomean(1/ttft.p50, 1/tpot.p50, req/s) | burst | geomean obj ≈ `geomean(1/0.2123, 1/0.02506, 0.0382)` | 100% success |

Public reference snapshot (H100, 2 h vLLM, from program.md, 2026-05-21):
SMAC3 11.53x aggregate (A 4.37x, B 15.23x, C 46.70x, D geomean 5.69x); vLLM
default 4.05x aggregate (A 1.25x, B 2.25x, C 48.69x, D 1.96x).

## Current best valid launcher (this advisor branch)

| Scenario | Status | Best speedup_over_pytorch | Launcher | PR | W&B run |
|---|---|---|---|---|---|
| A | **winner merged** | **1.261x** (TTFT p50 0.348 s) | `senpai/launchers/A/vllm-prefill-a/start_server.sh` | #61 | `aucwenw1` |
| B | open | — (no candidate) | PyTorch torch backend baseline | — | — |
| C | open | — (no candidate) | PyTorch torch backend baseline | — | — |
| D | open | — (no candidate) | PyTorch torch backend baseline | — | — |

**Scenario A winner recipe (merged PR #61):**
vLLM 0.11.0, `--gpu-memory-utilization 0.90`, `--max-num-seqs 16`,
`--max-num-batched-tokens 16384`, `--enable-chunked-prefill`,
`--enable-prefix-caching`, `--kv-cache-dtype auto` (FP8 KV rejected by
FlashAttention on this Blackwell hardware), `VLLM_ATTENTION_BACKEND=FLASH_ATTN`.
Quality: MMLU-Pro 0.296 / 0.298 baseline, ratio 0.9933 (pass).

**Known hardware constraints (RTX PRO 6000 Blackwell, this pod):**
- `--kv-cache-dtype fp8`: REJECTED by FlashAttention on this device.
- `VLLM_ATTENTION_BACKEND=FLASHINFER`: JIT fails (CUDA toolkit header mismatch).
- `--max-model-len 131072`: REJECTED by vLLM 0.11 (Mistral config is 32768).
- Safe choices: `--kv-cache-dtype auto`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN`
  (or omit env var), `INFERENCE_BENCH_MAX_MODEL_LEN=32768` (set by runtime_env.sh).

The reference point to beat per scenario is the SMAC3 H100 snapshot from program.md.

## Update history

- 2026-05-24 16:52: PR #61 merged (r4-fern, Scenario A, vLLM prefill launcher).
  New best for Scenario A: 1.261x (TTFT p50 0.348 s, quality pass).
- 2026-05-24: ledger created. PyTorch baselines and request files imported from
  PVC; preflight passed.
