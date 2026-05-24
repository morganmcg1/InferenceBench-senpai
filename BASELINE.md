# InferenceBench SENPAI Baseline — ib-20260524-hardened-r1

Live advisor-owned ledger for this research tag. Updated when a terminal
review-ready PR becomes the new current best for its scenario.

- **Research tag:** `ib-20260524-hardened-r1`
- **Advisor branch:** `ib-20260524-hardened-r1`
- **Hardware setting:** RTX PRO 6000 Blackwell-class shakedown (NOT
  leaderboard-comparable; H100 confirmation required before paper claims)
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Time budget:** 2 hour InferenceBench window
- **PyTorch baseline source:** prepared scoring assets at
  `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`
  (seed 248; preflight passed 2026-05-24)

## PyTorch baseline raw objectives (RTX PRO 6000, seed 248)

| Scenario | Raw objective | Value | Notes |
|---|---|---:|---|
| A: TTFT | `1 / ttft.p50` (burst, c=1) | `2.281` /s | ttft.p50 = 0.4385 s on 8192-token prompts |
| B: TPOT | `1 / tpot.p50` (burst, c=1) | `39.76` /s | tpot.p50 = 0.0252 s on 8192-token outputs |
| C: req/s | geomean req/s across burst/poisson/constant | `~0.085` req/s | burst c=64, poisson 32, constant 16 (256 reqs each) |
| D: balanced | geomean(1/ttft, 1/tpot, req/s) burst c=4 | (composite) | ttft 0.212 s, tpot 0.0251 s, 0.0382 req/s |

`speedup_over_pytorch = candidate_raw_objective / pytorch_baseline_raw_objective`.

## Current best launcher per scenario

| Scenario | Best launcher | Speedup vs PyTorch | Quality pass | PR | W&B run | Notes |
|---|---|---:|---|---|---|---|
| A | `senpai/launchers/scA/vllm-fp8-flashinfer-prefill/` | **1.884x** | PASS (ratio 1.000) | #57 | v051l94c | FLASH_ATTN + FP16 KV + FP8 weights. FlashInfer/FP8 KV blocked on Blackwell RTX 6000. |
| B | _none yet_ | `1.00x` | n/a | — | — | first round |
| C | _none yet_ | `1.00x` | n/a | — | — | first round |
| D | _none yet_ | `1.00x` | n/a | — | — | first round |

### Scenario A best — detail (2026-05-24, PR #57)

- **Launcher:** `senpai/launchers/scA/vllm-fp8-flashinfer-prefill/start_server.sh`
- **Effective flags:** `--quantization fp8 --kv-cache-dtype auto --max-num-seqs 16 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.92 --block-size 16 --no-enable-chunked-prefill --enable-prefix-caching`; `VLLM_ATTENTION_BACKEND=FLASH_ATTN` (env); `VLLM_USE_FLASHINFER_SAMPLER=0`; `VLLM_DISABLE_FLASHINFER_PREFILL=1`
- **Key metrics:**
  - `scenario/A/speedup_over_pytorch` = **1.884x**
  - `ttft.p50` = 0.233 s (was 0.4385 s), `ttft.p90/p99` = 0.262 / 0.271 s
  - `tpot.p50` = 0.01792 s (1.44x faster than baseline)
  - `request_throughput` = 0.0889 req/s
  - `quality/mmlu_pro_observed_accuracy` = 0.298 (ratio 1.000 — **PASS**)
  - `failure_count` = 0 / 128; `vram_peak_mb` = 92,771
- **W&B run:** [v051l94c](https://wandb.ai/wandb-applied-ai-team/inferencebench-senpai/runs/v051l94c) group `scA-round1`
- **Reproduce:**
  ```bash
  python senpai/create_task_workspace.py --scenario A --output /tmp/ib-scA --starting-point vllm_running
  cp senpai/launchers/scA/vllm-fp8-flashinfer-prefill/start_server.sh /tmp/ib-scA/task/start_server.sh
  cd /tmp/ib-scA/task && source ./eval_env.sh && ./clean_eval_artifacts.sh && \
    ./test_server.sh > agent/server.log 2>&1 & \
    python evaluate.py --json-output-file metrics_full.json
  ```
- **RTX 6000 hardware notes:** FlashInfer attention at runtime fails (`_sm_scale` assertion in vLLM 0.11 / FlashInfer 0.6.11 on Blackwell SM 120f). FlashAttention rejects `--kv-cache-dtype fp8` on Blackwell. Use `VLLM_ATTENTION_BACKEND=FLASH_ATTN` + `--kv-cache-dtype auto` for reliable startup.

## Update history

- 2026-05-24 (initial) — branch bootstrapped from
  `codex/inferencebench-senpai-target` with shared-pod hardening and prepared
  RTX PRO 6000 scoring assets. No SENPAI candidates measured yet on this
  branch.
- 2026-05-24 17:11 — PR #57 merged (r1-tanjiro). Scenario A new best: 1.884x
  speedup. First winner on this branch. RTX 6000 compatibility constraint
  documented: no FlashInfer attention, no FP8 KV cache.
