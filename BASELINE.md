# SENPAI Baseline — `ib-20260528-scen-d-r1`

Live advisor-owned baseline ledger for the Scenario D parity launch.

## Setting

- **Scenario:** D (General). LongBench-v2 prompts, 4096 input target / 2048 output target, 96 requests, burst concurrency 4, temperature 0.4.
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`.
- **GPU:** NVIDIA RTX PRO 6000 Blackwell, ~96 GB VRAM (shakedown). Not leaderboard-comparable until repeated on H100.
- **Time budget:** ~2 hours total for the launch.
- **W&B:** `wandb-applied-ai-team/inferencebench-senpai`, group `ib-20260528-scen-d-r1`.

## PyTorch reference (imported from `/mnt/new-pvc/.../rtxpro6000-seed248`)

- `generation_throughput_tokens_per_s`: 38.21
- `request_throughput_req_per_s`: 0.0382
- `ttft.p50/p90/p99`: 0.21 / 0.25 / 0.79 s
- `tpot.p50/p90/p99`: 0.025 / 0.055 / 0.127 s
- 96/96 successful, burst profile only.

## H100 reference snapshot (program.md, 2026-05-21)

| Method | Sc. D speedup |
|---|---:|
| SMAC3 vLLM 2h | 5.69x |
| TPE vLLM 2h | 5.58x |
| Random vLLM 2h | 5.42x |
| Claude Sonnet 4.6 agent | 3.01x |
| vLLM default | 1.96x |
| SGLang default | 2.14x |

This is H100, not RTX PRO 6000. Used for *direction*, not for ranking the current launch.

## Current best (live)

| Rank | PR | Engine | Recipe slug | Quick `speedup_over_pytorch` | Full `speedup_over_pytorch` | Quality ratio | W&B run | Notes |
|---|---|---|---|---|---|---|---|---|
| 1 | #155 (merged) | vLLM | frieren-vllm-balanced | 1.25x | **1.2919x** | 1.020 (PASS, n=500) | `lyzskqbu` | Chunked-prefill ON, prefix-caching ON, max-num-seqs=64, max-num-batched-tokens=16384, block-size=16, kv-cache-dtype=auto, gpu-mem-util=0.92, CUDA graphs ON, FLASH_ATTN. RTX PRO 6000 Blackwell, seed 248. |

**Primary metric to beat:** `scenario/D/speedup_over_pytorch` = **1.2919x** (geomean raw = 2.4931).

## Quick / provisional ledger

| PR | Student | Engine | Quick speedup | Notes |
|---|---|---|---|---|
| #155 | scen-d-frieren | vLLM tuned | 1.25x (Arm 2) | Promoted to full → 1.2919x → **merged** |
| #157 | scen-d-fern | SGLang LPM | 1.168x (Arm 1) | c=1, triton attn +59% gen throughput. Pivoting to speculative arm. |

## Failed / closed

(empty)

## 2026-05-28 17:46 — PR #155 merged: vLLM tuned balanced (frieren)

- **speedup_over_pytorch (full, n=96):** 1.2919x
- **geomean raw:** 2.4931 (baseline 1.9298)
- **generation_throughput:** 56.03 tok/s (baseline 38.21 tok/s, +46.6%)
- **request_throughput:** 0.04581 req/s (baseline 0.0382 req/s)
- **TTFT p50/p90/p99:** 0.1746 / 0.1958 / 1.2126 s (p99 regresses: chunked-prefill interleave)
- **TPOT p50/p90/p99:** 0.01694 / 0.02674 / 0.04802 s (p50 1.48x faster, p90 2.06x, p99 2.64x)
- **Quality:** MMLU-Pro n=500, observed=0.304, baseline=0.298, ratio=1.020 ≥ tau=0.95 → PASS
- **Speed:** 96/96 success, 0 failures
- **W&B:** `lyzskqbu` (group `ib-20260528-scen-d-r1`)
- **Launcher:** `senpai/launchers/D/frieren-vllm-balanced/start_server.sh`
- **Reproduce:**
  ```bash
  cd target/ && python senpai/create_task_workspace.py --scenario D \
    --output /tmp/inferencebench-scenario-d --starting-point vllm_running
  # start_server.sh: vLLM, --max-num-seqs 64 --max-num-batched-tokens 16384
  # --enable-chunked-prefill --enable-prefix-caching --block-size 16
  # --kv-cache-dtype auto --gpu-memory-utilization 0.92 --max-model-len 32768
  python evaluate.py --scenario D --full
  ```
- **Analysis:** TPOT distribution tightened dramatically (CUDA graphs + wider in-flight batch). TTFT improved 1.22x median; p99 regressed due to chunked-prefill interleaving 4096-token inputs. Quality slightly above baseline. Eval effectively ran at c=1 (harness bug at runner.py:1159 — see frieren's notes for details). Speedup is apples-to-apples vs PyTorch baseline (also c=1).

## Update history

- 2026-05-28 16:36: Initial scaffold created. Preflight passes for Scenario D. Two students idle.
- 2026-05-28 17:46: PR #155 merged. New best: 1.2919x speedup_over_pytorch. Frieren vLLM tuned.
