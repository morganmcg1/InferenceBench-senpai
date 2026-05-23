# SENPAI BASELINE — InferenceBench ib-20260523-rerun-r3

Live advisor-owned baseline ledger. Update when a terminal review-ready PR with
a clean relaunch beats the current best on its scenario.

## Setting

- Research tag: `ib-20260523-rerun-r3`
- Advisor branch: `ib-20260523-rerun-r3-advisor`
- Base model: `mistralai/Mistral-7B-Instruct-v0.3`
- Hardware: NVIDIA RTX PRO 6000 Blackwell ~96GB (shakedown). H100 leaderboard
  numbers in `target/program.md` reference table are public references, NOT a
  live comparison.
- Per-run wall-clock budget: `SENPAI_TIMEOUT_MINUTES` (per training run).
  Research-program budget: 2 hours total.
- W&B: `wandb-applied-ai-team/inferencebench-senpai`
- PyTorch speed baseline: produced per-pod by
  `senpai/preflight.py --scenario all --expected-gpu "RTX PRO 6000" --require-wandb`.
  Each student must run preflight before reporting `speedup_over_pytorch`.

## Starting launcher

`src/starting_points/vllm_running/start_server.sh` — vLLM defaults with
`--gpu-memory-utilization 0.90 --trust-remote-code --disable-log-stats`,
`max_model_len=131072`, no chunked-prefill, no prefix caching, no quantization,
no CUDA graphs override.

## Current best per scenario

No SENPAI launcher has been measured on this advisor branch yet. Current best
is `vllm_running` defaults. The first terminal review-ready PR that beats
PyTorch baseline on a scenario becomes the live current best for that scenario.

| Scenario | Primary metric | Current best `speedup_over_pytorch` | Launcher path | W&B run | Notes |
|---|---|---:|---|---|---|
| A: Input-heavy (`1/ttft.p50` burst) | `scenario/A/speedup_over_pytorch` | TBD (vLLM default) | `src/starting_points/vllm_running/start_server.sh` | — | Awaiting first measured result on RTX PRO 6000 |
| B: Output-heavy (`1/tpot.p50` burst) | `scenario/B/speedup_over_pytorch` | TBD (vLLM default) | `src/starting_points/vllm_running/start_server.sh` | — | Awaiting first measured result on RTX PRO 6000 |
| C: High-load (req/s geomean) | `scenario/C/speedup_over_pytorch` | TBD (vLLM default) | `src/starting_points/vllm_running/start_server.sh` | — | Awaiting first measured result on RTX PRO 6000 |
| D: General (geomean) | `scenario/D/speedup_over_pytorch` | TBD (vLLM default) | `src/starting_points/vllm_running/start_server.sh` | — | Awaiting first measured result on RTX PRO 6000 |
| Aggregate (cross-scenario confirmation) | `aggregate/geomean_speedup_over_pytorch` | TBD | — | — | Run only after a mature single-scenario winner. |

## Public reference snapshot (H100, 2026-05-21, do not treat as live)

| Method | Agg | A TTFT | B TPOT | C req/s | D geomean |
|---|---:|---:|---:|---:|---:|
| SMAC3 search, 2h vLLM | 11.53x | 4.37x | 15.23x | 46.70x | 5.69x |
| TPE search, 2h vLLM | 11.25x | 4.48x | 14.76x | 43.46x | 5.58x |
| Random, 2h vLLM | 10.20x | 4.21x | 11.34x | 41.81x | 5.42x |
| Claude Sonnet 4.6 best | 8.08x | 3.47x | 12.03x | 33.93x | 3.01x |
| vLLM default | 4.05x | 1.25x | 2.25x | 48.69x | 1.96x |

These ratios come from H100. RTX PRO 6000 shakedown ratios will likely differ,
so use them only as *direction* signal until repeated on H100.

## Round 1 policy: raw-objective primary metrics

PyTorch speed/quality baselines do not yet exist on this pod's writable
filesystem for any scenario. Generating them inside the 2 h program
budget would consume the GPU time we need for the actual launcher
experiments. For round 1 only, terminal results report **raw scenario
objectives** as `primary_metric.name`:

- Scenario A → `scenario/A/raw/inverse_ttft_p50`
- Scenario B → `scenario/B/raw/inverse_tpot_p50`
- Scenario C → `scenario/C/raw/request_throughput_req_per_s_geomean`
- Scenario D → `scenario/D/raw/geomean_inverse_latency_throughput`

These are research-grade signals, NOT leaderboard-comparable. The
`speedup_over_pytorch` columns above stay TBD until a future tooling PR
generates the PyTorch baselines.

## Update history

- 2026-05-23 09:50: file created. Awaiting first measured launcher.
- 2026-05-23 10:24: round 1 raw-objective pivot recorded. Reasoning:
  PyTorch baselines absent on the pod, 2 h budget can't absorb baseline
  generation. r3-fern fixed vLLM 0.11→0.21 install on this pod (huge
  unblock). r3-tanjiro patched a BPE round-trip drift in
  `senpai/materialize_requests.py`. Slot order reshuffled: tanjiro →
  fern → frieren (frieren still silent).
