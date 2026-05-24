# InferenceBench Live Baseline — ib-20260524-leasefix-r5

This is the live advisor-owned baseline ledger for this SENPAI run. Update on
every terminal review-ready PR that beats the current best.

## Setting
- Base model: `mistralai/Mistral-7B-Instruct-v0.3`
- Hardware (current shakedown): NVIDIA RTX PRO 6000 Blackwell ~96 GB
- Target hardware (leaderboard): NVIDIA H100 80GB (results not directly comparable yet)
- Budget: 2 hours total per run
- Branch: `ib-20260524-leasefix-r5`
- W&B: `wandb-applied-ai-team/inferencebench-senpai`

## Starting launcher
- `src/starting_points/vllm_running/start_server.sh` — vLLM default with
  `--gpu-memory-utilization 0.90`, `--trust-remote-code`, `--disable-log-stats`,
  and no chunked-prefill / KV-FP8 / speculative-decoding flags set.
- Runtime helper `senpai/runtime_env.sh` defaults
  `INFERENCE_BENCH_MAX_MODEL_LEN=32768` and disables the implicit FlashInfer
  sampler/prefill path on RTX PRO 6000 unless a launcher explicitly opts back
  in.

## Current best per scenario (RTX PRO 6000 shakedown)

| Scenario | Primary metric | Current best | Launcher | PR | W&B | Notes |
|---|---|---:|---|---|---|---|
| A — input-heavy (TTFT)   | `scenario/A/speedup_over_pytorch` | **1.8903x** | `senpai/launchers/A/big-prefill-fp8/start_server.sh` | #78 | [9hvhqa1u](https://wandb.ai/wandb-applied-ai-team/inferencebench-senpai/runs/9hvhqa1u) | FP8 weights + `--max-num-batched-tokens 16384 --no-enable-chunked-prefill --max-num-seqs 32 --gpu-memory-utilization 0.92 --max-model-len 16384`; MMLU-Pro ratio **1.000** (PASS), `ttft.p50 = 232 ms` |
| B — output-heavy (TPOT)  | `scenario/B/speedup_over_pytorch` | _no measurement yet_ | _vLLM default starting point_ | — | — | Largest headroom on H100 ref (15x SMAC3 vs 2.25x default) |
| C — high-load (req/s)    | `scenario/C/speedup_over_pytorch` | _no measurement yet_ | _vLLM default starting point_ | — | — | Default already near top on H100 ref |
| D — general (geomean)    | `scenario/D/speedup_over_pytorch` | **1.5015x** | `senpai/launchers/D/balanced-fp8/start_server.sh` | #82 | [85zn8jd4](https://wandb.ai/wandb-applied-ai-team/inferencebench-senpai/runs/85zn8jd4) | FP8 weights + chunked prefill + max-num-seqs 64 + max-num-batched-tokens 8192; MMLU-Pro ratio 0.9933 (PASS at tau=0.95) |

## Public H100 reference snapshot (2026-05-21)

Headline H100 numbers from `program.md` for context only; RTX PRO 6000
shakedown results are not directly comparable.

| Method | Aggregate | Sc. A TTFT | Sc. B TPOT | Sc. C req/s | Sc. D geomean |
|---|---:|---:|---:|---:|---:|
| SMAC3 search, 2h vLLM | 11.53x | 4.37x | 15.23x | 46.70x | 5.69x |
| TPE search, 2h vLLM | 11.25x | 4.48x | 14.76x | 43.46x | 5.58x |
| Random search, 2h vLLM | 10.20x | 4.21x | 11.34x | 41.81x | 5.42x |
| Best agent (Claude Sonnet 4.6) | 8.08x | 3.47x | 12.03x | 33.93x | 3.01x |
| vLLM default, no agent | 4.05x | 1.25x | 2.25x | 48.69x | 1.96x |
| PyTorch baseline | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |

## Update history
- 2026-05-24: file created. No live measurements yet.
- 2026-05-24 23:08 UTC: PR #82 merged — Sc. D = **1.5015x** speedup_over_pytorch (tanjiro, balanced-fp8 launcher). First on-hardware measurement of this run. MMLU-Pro ratio 0.9933 over n=500, 0.7% margin above tau=0.95. W&B run `85zn8jd4`.
- 2026-05-24 23:34 UTC: PR #78 merged — Sc. A = **1.8903x** speedup_over_pytorch (fern, big-prefill-fp8 launcher). MMLU-Pro ratio exactly **1.000** at n=500 (observed 0.298 == baseline 0.298). One-shot prefill (`--max-num-batched-tokens 16384 --no-enable-chunked-prefill`) is the dominant lever at concurrency=1. W&B run `9hvhqa1u`.
