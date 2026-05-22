# InferenceBench SENPAI Baseline Ledger

Advisor branch: `ib-20260522-r3-advisor`
Research tag: `ib-20260522-r3`
W&B: `wandb-applied-ai-team/inferencebench-senpai`
Base model: `mistralai/Mistral-7B-Instruct-v0.3`
Hardware: 1x NVIDIA H100 80GB (shared across 3 packed students)
Time budget: 2 hours total

## Live current bests (this advisor branch)

Speedups are measured vs. the PyTorch naive baseline using the official
`evaluate.py --json-output-file` output and computed by
`senpai/summarize_metrics.py`. Quality gate must pass (MMLU-Pro ratio >= 0.95).

| Scenario | Primary metric | Current best (this branch) | Launcher | W&B run | PR |
|---|---|---:|---|---|---|
| A: Input heavy (TTFT) | `scenario/A/speedup_over_pytorch` | (none yet) | - | - | - |
| B: Output heavy (TPOT) | `scenario/B/speedup_over_pytorch` | (none yet) | - | - | - |
| C: High load (req/s geomean) | `scenario/C/speedup_over_pytorch` | (none yet) | - | - | - |
| D: General (geomean) | `scenario/D/speedup_over_pytorch` | (none yet) | - | - | - |
| Aggregate (A-D geomean) | `aggregate/geomean_speedup_over_pytorch` | (none yet) | - | - | - |

## Public reference snapshot (program.md, 2026-05-21)

This is the target to beat. Numbers are speedup over PyTorch on the same
Mistral-7B-Instruct-v0.3 / H100 80GB / 2h setting.

| Method | Aggregate | A TTFT | B TPOT | C req/s | D geomean |
|---|---:|---:|---:|---:|---:|
| SMAC3 search, 2h vLLM | 11.53x | 4.37x | 15.23x | 46.70x | 5.69x |
| TPE search, 2h vLLM | 11.25x | 4.48x | 14.76x | 43.46x | 5.58x |
| Random search, 2h vLLM | 10.20x | 4.21x | 11.34x | 41.81x | 5.42x |
| Best agent (Claude Sonnet 4.6) | 8.08x | 3.47x | 12.03x | 33.93x | 3.01x |
| vLLM default | 4.05x | 1.25x | 2.25x | 48.69x | 1.96x |
| SGLang default | 3.92x | 1.22x | 1.77x | 51.12x | 2.14x |
| HF TGI default | 3.30x | 1.14x | 1.37x | 41.94x | 1.80x |
| PyTorch baseline | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |

## Update history

- 2026-05-22 — Initial ledger created. No terminal launchers yet; assigning
  first round of hypotheses to r3-frieren (Scenario B), r3-fern (Scenario C),
  r3-tanjiro (Scenario A) sharing a single H100.
