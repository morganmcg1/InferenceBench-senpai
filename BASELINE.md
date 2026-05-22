# InferenceBench Live Baseline — ib-20260522-r4

- **Target model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Hardware:** 1x NVIDIA H100 80GB
- **Time budget:** 2 hours per benchmark run; ~2 hours per SENPAI program
- **W&B project:** `wandb-applied-ai-team/inferencebench-senpai`
- **Advisor branch:** `ib-20260522-r4-advisor`
- **Research tag:** `ib-20260522-r4`

The paper-facing metric is **speedup over PyTorch baseline**. Higher is better.
Each scenario reports the speedup for its scenario primary objective; aggregate
is the geomean across A-D.

## Current best (live)

No SENPAI experiment has been completed yet on this advisor branch. The values
below come from the public reference snapshot in `program.md` (timestamp:
2026-05-21). They are the targets a new launcher must beat to be a winner.

| Scenario | Primary speedup metric                          | Current best | Source (PR / reference) |
|----------|--------------------------------------------------|--------------|--------------------------|
| A        | `scenario/A/speedup_over_pytorch` (1/ttft.p50)   | **4.37x**    | SMAC3 search ref (2026-05-21) |
| B        | `scenario/B/speedup_over_pytorch` (1/tpot.p50)   | **15.23x**   | SMAC3 search ref (2026-05-21) |
| C        | `scenario/C/speedup_over_pytorch` (req/s geomean)| **51.12x**   | SGLang default ref (2026-05-21) |
| D        | `scenario/D/speedup_over_pytorch` (geomean burst)| **5.69x**    | SMAC3 search ref (2026-05-21) |
| Aggregate| `aggregate/geomean_speedup_over_pytorch`         | **11.53x**   | SMAC3 search ref (2026-05-21) |

> Note: Scenario C's leader is the **SGLang default** (51.12x), not SMAC3
> (46.70x). The continuous-batching scheduler in SGLang is already extremely
> strong on throughput-bounded workloads. Treat 51.12x as the hard line to beat
> on C; treat 48.69x (vLLM default) as the strongest vLLM-side target.

## Public reference snapshot (informational only)

These are the original `program.md` reference numbers and serve as ranking
context. Update only the "Current best (live)" rows above as SENPAI winners are
merged into this advisor branch.

| Method                                      | Aggregate | Sc. A | Sc. B | Sc. C | Sc. D |
|---------------------------------------------|----------:|------:|------:|------:|------:|
| SMAC3 search, 2h vLLM                       |   11.53x  | 4.37x |15.23x |46.70x | 5.69x |
| TPE search, 2h vLLM                         |   11.25x  | 4.48x |14.76x |43.46x | 5.58x |
| Random search, 2h vLLM                      |   10.20x  | 4.21x |11.34x |41.81x | 5.42x |
| Best listed agent (Claude Sonnet 4.6)       |    8.08x  | 3.47x |12.03x |33.93x | 3.01x |
| vLLM default, no agent                      |    4.05x  | 1.25x | 2.25x |48.69x | 1.96x |
| SGLang default, no agent                    |    3.92x  | 1.22x | 1.77x |51.12x | 2.14x |
| HF TGI default, no agent                    |    3.30x  | 1.14x | 1.37x |41.94x | 1.80x |
| PyTorch baseline                            |    1.00x  | 1.00x | 1.00x | 1.00x | 1.00x |

## Update history

- 2026-05-22 — Advisor created live baseline doc; no SENPAI runs yet.
