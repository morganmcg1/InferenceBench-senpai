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

## Current Best Launcher (this branch)

No launcher experiments completed yet. The default reference is the
`vllm_running` starting point (`src/starting_points/vllm_running/start_server.sh`):
vLLM defaults with `--gpu-memory-utilization 0.90 --max-model-len 131072
--trust-remote-code --disable-log-stats`.

| Scenario | Best speedup_over_pytorch | PR | Launcher | W&B run | Notes |
|---|---:|---|---|---|---|
| A | unmeasured | – | starting_points/vllm_running | – | first measurement pending |
| B | unmeasured | – | starting_points/vllm_running | – | first measurement pending |
| C | unmeasured | – | starting_points/vllm_running | – | first measurement pending |
| D | unmeasured | – | starting_points/vllm_running | – | first measurement pending |

## Public Reference (H100, 2h budget, for direction only — NOT comparable on RTX PRO 6000)

| Method | Aggregate | A | B | C | D |
|---|---:|---:|---:|---:|---:|
| SMAC3 search 2h vLLM | 11.53x | 4.37x | 15.23x | 46.70x | 5.69x |
| TPE search 2h vLLM | 11.25x | 4.48x | 14.76x | 43.46x | 5.58x |
| vLLM default | 4.05x | 1.25x | 2.25x | 48.69x | 1.96x |
| PyTorch | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |

## Update History

- 2026-05-24 — initialized advisor ledger; preflight passed via imported PVC
  assets; first round of student PRs in flight (Scenario B speculative
  decoding, Scenario A chunked-prefill + FlashInfer, Scenario D balanced
  tuning).
