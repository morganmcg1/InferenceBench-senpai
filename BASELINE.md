# SENPAI Inference Baselines — ib-20260527-latest3-r1

- **Advisor branch:** ib-20260527-latest3-r1
- **Tag:** ib-20260527-latest3-r1
- **Target repo:** morganmcg1/InferenceBench-senpai
- **Base model:** mistralai/Mistral-7B-Instruct-v0.3
- **GPU:** RTX PRO 6000 Blackwell ~96GB VRAM (shakedown; not leaderboard-comparable to H100)
- **Scoring seed:** 248 (imported from `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`)
- **W&B project:** `wandb-applied-ai-team/inferencebench-senpai`
- **Time budget:** 2 hour run window
- **Starting point:** vLLM default launcher `src/starting_points/vllm_running/start_server.sh`
- **Preflight:** PASS (require_scoring_preflight.sh, all scenarios, RTX PRO 6000)

## PyTorch Baselines (denominator for speedup)

| Scenario | Workload | Raw objective | PyTorch baseline value | Requests |
|---|---|---|---:|---:|
| A: Input-heavy   | 8192 in / 1024 out, burst c=1   | `1 / ttft.p50` (burst)                                       | 2.281 (ttft.p50=0.4385s) | 128/128 |
| B: Output-heavy  | 1024 in / 8192 out, burst c=1   | `1 / tpot.p50` (burst)                                       | 39.69 (tpot.p50=0.02519s) | 64/64 |
| C: High-load     | 1024 in / 1024 out, 3 profiles  | geomean(`request_throughput_req_per_s`) over burst/poisson/constant | 0.0847 req/s | 768/768 |
| D: General       | 4096 in / 2048 out, burst c=4   | geomean(`1/ttft.p50`, `1/tpot.p50`, `req_thr`) (burst)       | 1.928 | 96/96 |

Speedup is `candidate_raw_objective / pytorch_baseline_raw_objective`. Higher is better. Quality gate is mandatory (`mmlu_pro.ratio >= 0.95`).

## Reference Snapshot (public H100, 2026-05-21)

For direction only. RTX PRO 6000 shakedown numbers are not leaderboard-comparable.

| Method | Sc. A TTFT | Sc. B TPOT | Sc. C req/s | Sc. D geomean | Aggregate |
|---|---:|---:|---:|---:|---:|
| SMAC3 2h vLLM (best public) | 4.37x | 15.23x | 46.70x | 5.69x | 11.53x |
| vLLM default | 1.25x | 2.25x | 48.69x | 1.96x | 4.05x |
| SGLang default | 1.22x | 1.77x | 51.12x | 2.14x | 3.92x |
| PyTorch baseline | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |

## Current Best (terminal, full-eval, validated)

| Scenario | Speedup over PyTorch | Launcher | W&B run | PR |
|---|---:|---|---|---|
| A | — | none | — | — |
| B | — | none | — | — |
| C | — | none | — | — |
| D | — | none | — | — |

No terminal full-eval winners yet on this advisor branch.

## Provisional / Quick / Unconfirmed Candidates

(none yet)

## Failed Launches / Dead Ends

(none yet)

## Update History

- 2026-05-27 — initial ledger created. Preflight PASS for RTX PRO 6000 seed248 scoring assets. Starting search from vLLM default launcher.
