# InferenceBench SENPAI Baseline Ledger

- Research tag: `ib-20260527-lean1-r1`
- Advisor branch: `ib-20260527-lean1-r1`
- Base model: `mistralai/Mistral-7B-Instruct-v0.3`
- Hardware: NVIDIA RTX PRO 6000 Blackwell (~96 GB VRAM). Shakedown only, not
  leaderboard-comparable to the H100 80GB paper setting.
- Time budget: 2 hours per SENPAI run.
- Starting launcher: `src/starting_points/vllm_running/start_server.sh`.
- PyTorch baseline metrics source: official `src/eval/inference/baselines/speed/torch/*` (seed=248) imported from PVC
  `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`.
- W&B project: `wandb-applied-ai-team/inferencebench-senpai`.

## Reference Snapshot (H100, paper, not measured here)

| Method | Sc. A TTFT | Sc. B TPOT | Sc. C req/s | Sc. D geomean | Aggregate |
|---|---:|---:|---:|---:|---:|
| SMAC3 search, 2h vLLM | 4.37x | 15.23x | 46.70x | 5.69x | 11.53x |
| vLLM default, no agent | 1.25x | 2.25x | 48.69x | 1.96x | 4.05x |
| SGLang default, no agent | 1.22x | 1.77x | 51.12x | 2.14x | 3.92x |
| PyTorch baseline | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |

These are H100 paper numbers, included only to anchor expectations. RTX PRO 6000
measurements live in the per-scenario tables below once they exist.

## Current Best (measured on this hardware)

| Scenario | Speedup over PyTorch | Launcher | W&B run | PR | Notes |
|---|---:|---|---|---|---|
| A | **1.893x** | `senpai/launchers/A/chunked-prefill-fp8/start_server.sh` | u44zjwyh | #126 | RTX PRO 6000, seed=248; quality_ratio=1.0 (n=500); 128/128 speed requests; merged 2026-05-27 17:17 |
| B | _unset_ | _unset_ | _unset_ | _unset_ | fern full eval in progress |
| C | **25.62x** | `senpai/launchers/C/fp8-large-batch-prefix-cache/start_server.sh` | ckfmuinz | #128 | RTX PRO 6000, seed=248; quality_ratio=1.0 (n=500); 768/768 speed requests; merged 2026-05-27 16:46 |
| D | _unset_ | _unset_ | _unset_ | _unset_ | not assigned yet |

## Quick / Provisional Probes (research signals, not benchmark wins)

| Student | PR | Scenario | Quick speedup | W&B run | Notes |
|---|---|---|---:|---|---|
| frieren | #126 | A | 1.91x | u2dxy31b | chunked prefill + FP8 weights; quality n=16 screen only |
| tanjiro | #128 | C | 2.76x geomean | ocz25mgd | cold-start burst dominated; steady-state 3.86x; full confirmed |

## Update History

- 2026-05-27 16:14 UTC — ledger initialized; preflight against `rtxpro6000-seed248` passed.
- 2026-05-27 16:46 UTC — **Scenario C** first winner: PR #128 (tanjiro), 25.62x speedup over PyTorch, quality 1.0, W&B `ckfmuinz`. Launcher: vLLM FP8 weights + max-num-seqs=256 + max-num-batched-tokens=16384 + chunked-prefill + prefix-caching + gpu-mem-util=0.92.
- 2026-05-27 17:17 UTC — **Scenario A** first winner: PR #126 (frieren), 1.893x speedup over PyTorch, quality ratio 1.0 (n=500), W&B `u44zjwyh`. Launcher: vLLM FP8 weights + chunked-prefill + max-num-batched-tokens=16384 + max-num-seqs=16 + gpu-mem-util=0.90. TTFT p50=0.232s vs PyTorch 0.439s.
