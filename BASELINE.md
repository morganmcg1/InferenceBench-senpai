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
| A | _unset_ | _unset_ | _unset_ | _unset_ | awaiting first valid full eval |
| B | _unset_ | _unset_ | _unset_ | _unset_ | awaiting first valid full eval |
| C | _unset_ | _unset_ | _unset_ | _unset_ | awaiting first valid full eval |
| D | _unset_ | _unset_ | _unset_ | _unset_ | not assigned yet this run |

## Quick / Provisional Probes (research signals, not benchmark wins)

_(updated as students post `terminal=false` checkpoints)_

## Update History

- 2026-05-27 16:14 UTC — ledger initialized at start of `ib-20260527-lean1-r1` run; preflight against
  `rtxpro6000-seed248` PVC assets passed.
