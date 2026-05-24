# InferenceBench SENPAI — Live Baseline (ib-20260524-ready-r4)

Live advisor-owned baseline ledger for the active SENPAI run. PRs that beat the
current best on a given scenario update that scenario row and append a history
entry.

- **Research tag:** `ib-20260524-ready-r4`
- **Advisor branch:** `ib-20260524-ready-r4-advisor`
- **Target base:** `codex/inferencebench-senpai-target`
- **W&B project:** `wandb-applied-ai-team/inferencebench-senpai`
- **Hardware:** 1× NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM) — shakedown only, not leaderboard-comparable
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Wall clock:** 2 hour total program budget
- **PyTorch baseline source:** `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248` (seed 248, all four scenarios passed preflight)

## Current best valid launcher per scenario

| Scenario | Primary metric | PyTorch baseline raw | Current best speedup | Launcher | W&B run | PR | Notes |
|---|---|---:|---:|---|---|---|---|
| A: input-heavy | `scenario/A/speedup_over_pytorch` | `1/ttft.p50 = 2.281 /s` (ttft.p50=0.4385s) | — | none yet | — | — | starting point: `src/starting_points/vllm_running/start_server.sh` |
| B: output-heavy | `scenario/B/speedup_over_pytorch` | `1/tpot.p50 = 39.76 /s` (tpot.p50=0.02515s) | — | none yet | — | — | starting point: `src/starting_points/vllm_running/start_server.sh` |
| C: high-load | `scenario/C/speedup_over_pytorch` | geomean req/s burst+poisson+constant ≈ 0.0847 req/s | — | none yet | — | — | starting point: `src/starting_points/vllm_running/start_server.sh` |
| D: general | `scenario/D/speedup_over_pytorch` | geomean(1/ttft.p50, 1/tpot.p50, req/s) of burst ≈ 0.836 | — | none yet | — | — | starting point: `src/starting_points/vllm_running/start_server.sh` |

## Public reference snapshot (H100 80GB, 2h, Mistral-7B-Instruct-v0.3)

For directional context only. RTX PRO 6000 shakedown results are **not**
leaderboard-comparable until repeated on H100.

| Method | Sc. A | Sc. B | Sc. C | Sc. D | Aggregate |
|---|---:|---:|---:|---:|---:|
| SMAC3 search, 2h vLLM | 4.37x | 15.23x | 46.70x | 5.69x | 11.53x |
| TPE search, 2h vLLM | 4.48x | 14.76x | 43.46x | 5.58x | 11.25x |
| Best listed agent (Sonnet 4.6) | 3.47x | 12.03x | 33.93x | 3.01x | 8.08x |
| vLLM default | 1.25x | 2.25x | 48.69x | 1.96x | 4.05x |
| SGLang default | 1.22x | 1.77x | 51.12x | 2.14x | 3.92x |

## Update history

- 2026-05-24 — initialized live baseline ledger. RTX PRO 6000 scoring assets
  preflighted at `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`
  for all four scenarios. No SENPAI launcher results yet; starting from vLLM
  default.
