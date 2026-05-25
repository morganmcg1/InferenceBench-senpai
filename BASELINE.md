# InferenceBench SENPAI Live Baseline

- **Research tag:** `ib-20260525-one4-r1`
- **Advisor branch:** `ib-20260525-one4-r1`
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Hardware:** NVIDIA RTX PRO 6000 Blackwell (~96 GB VRAM) — shakedown only, not leaderboard-comparable to H100 80 GB.
- **Dataset seed:** 248 (`rtxpro6000-seed248`)
- **MMLU-Pro PyTorch quality baseline:** observed accuracy `0.298`, gate `tau = 0.95`, so observed must be `>= 0.2831`.
- **Speed budget:** 2-hour total window for the entire research program (assignment + iteration + review).

## PyTorch speed baselines (RTX PRO 6000, seed=248)

These are the raw objectives used in the `speedup_over_pytorch` denominator.

| Scenario | Profile | PyTorch raw objective | Component metric(s) |
|---|---|---:|---|
| A: Input-heavy (8192 in, 1024 out, 128 req, burst c=1) | `burst` | `1/ttft.p50 = 2.281` | TTFT.p50 = 0.4385 s |
| B: Output-heavy (1024 in, 8192 out, 64 req, burst c=1) | `burst` | `1/tpot.p50 = 39.756` | TPOT.p50 = 0.02515 s |
| C: High-load (1024 in, 1024 out, 256 req × 3) | geomean(`burst`, `poisson`, `constant`) | `req/s = 0.0847` | req/s = 0.0845, 0.0847, 0.0849 |
| D: General (4096 in, 2048 out, 96 req, burst c=4) | `burst` | `geomean(1/ttft, 1/tpot, req/s) = 1.832` | TTFT 0.2123, TPOT 0.02506, req/s 0.0382 |

## Current best valid launcher per scenario

No SENPAI launcher has been measured yet on this branch. All entries below are
**open** — each scenario's current best is the PyTorch baseline at `1.00x` and
any improving candidate is mergeable.

| Scenario | Best launcher | Primary metric | Speedup over PyTorch | W&B run | PR |
|---|---|---|---:|---|---|
| A | — (PyTorch baseline) | `scenario/A/speedup_over_pytorch` | `1.00x` | — | — |
| B | — (PyTorch baseline) | `scenario/B/speedup_over_pytorch` | `1.00x` | — | — |
| C | — (PyTorch baseline) | `scenario/C/speedup_over_pytorch` | `1.00x` | — | — |
| D | — (PyTorch baseline) | `scenario/D/speedup_over_pytorch` | `1.00x` | — | — |
| Aggregate | — | `aggregate/geomean_speedup_over_pytorch` | `1.00x` | — | — |

## Public reference snapshot (2026-05-21, H100 80 GB)

Context only. Not directly comparable to RTX PRO 6000 results, but useful to
gauge headroom against well-tuned SMAC3/TPE search budgets.

| Method | Aggregate | A TTFT | B TPOT | C req/s | D geomean |
|---|---:|---:|---:|---:|---:|
| SMAC3 2h vLLM | 11.53x | 4.37x | 15.23x | 46.70x | 5.69x |
| TPE 2h vLLM | 11.25x | 4.48x | 14.76x | 43.46x | 5.58x |
| vLLM default | 4.05x | 1.25x | 2.25x | 48.69x | 1.96x |
| SGLang default | 3.92x | 1.22x | 1.77x | 51.12x | 2.14x |
| TGI default | 3.30x | 1.14x | 1.37x | 41.94x | 1.80x |

Largest known headroom over framework defaults:
- **B (output-heavy):** 15.23x SMAC3 vs 2.25x vLLM default → ~6.8x reachable, mostly from speculative decoding, CUDA graphs, and KV-cache tuning.
- **A (input-heavy):** 4.37x vs 1.25x → ~3.5x reachable, mostly from chunked prefill, prefix caching, and batched-token sizing.
- **D (general):** 5.69x vs 1.96x → ~2.9x reachable.
- **C (high-load):** 46.70x vs 48.69x → defaults already near ceiling; small gains only.

## Update history

- 2026-05-25 — Initialized ledger from PyTorch baselines on RTX PRO 6000
  seed=248. No SENPAI candidates measured yet.
