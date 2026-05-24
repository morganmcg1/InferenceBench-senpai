# InferenceBench SENPAI Baseline — `ib-20260524-hardened-r3`

Live advisor-owned ledger of the current best valid InferenceBench launcher per
scenario for this research tag. Update whenever a terminally-reviewed PR
becomes the new current best for a scenario.

## Active Setting

- Research tag: `ib-20260524-hardened-r3`
- Advisor branch: `ib-20260524-hardened-r3`
- Target base branch: `codex/inferencebench-senpai-target`
- Hardware: 1x NVIDIA RTX PRO 6000 Blackwell, ~96GB VRAM (SHAKEDOWN — not
  H100, not leaderboard-comparable)
- Base model: `mistralai/Mistral-7B-Instruct-v0.3`
- Time budget per training run: capped by `SENPAI_TIMEOUT_MINUTES`
- Whole-program budget: 2 hours (clock starts when SENPAI gate opens)
- Quality gate: MMLU-Pro 500-question subset, `tau=0.95` of PyTorch baseline
  accuracy (0.298)
- Scoring assets (preflight): imported from
  `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`
- W&B project: `wandb-applied-ai-team/inferencebench-senpai`

## PyTorch Baselines (raw objective per scenario, RTX PRO 6000, seed 248)

These are the denominator values used to compute `speedup_over_pytorch`:

| Scenario | Profile(s)                | Raw objective                              | PyTorch raw value |
|---------:|---------------------------|--------------------------------------------|------------------:|
| A        | burst (c=1)               | `1 / ttft.p50` (burst)                     | `1 / 0.4385 ≈ 2.281` |
| B        | burst (c=1)               | `1 / tpot.p50` (burst)                     | `1 / 0.02515 ≈ 39.756` |
| C        | burst / poisson / constant| geomean `request_throughput_req_per_s`     | `~0.0847 req/s`   |
| D        | burst (c=4)               | geomean(1/ttft.p50, 1/tpot.p50, req/s)     | derived           |

Quality baseline (PyTorch, MMLU-Pro 500@248): observed accuracy = 0.298. Gate
requires candidate observed accuracy ≥ 0.95 * 0.298 = 0.2831.

## Current Best Launcher (per scenario)

No SENPAI-validated launchers have been measured yet on this tag. Starting
point is `src/starting_points/vllm_running/start_server.sh` (vLLM default,
`--gpu-memory-utilization 0.90`, no quant, no spec decode, no chunked
prefill). It is not yet measured here; the first terminal student PRs will
populate the table.

| Scenario | PR  | Speedup vs PyTorch | MMLU-Pro acc. | W&B run | Launcher recipe |
|---------:|-----|--------------------:|--------------:|---------|-----------------|
| A        | —   | unmeasured          | —             | —       | (default vLLM) |
| B        | —   | unmeasured          | —             | —       | (default vLLM) |
| C        | —   | unmeasured          | —             | —       | (default vLLM) |
| D        | —   | unmeasured          | —             | —       | (default vLLM) |

## Reference Snapshot (2026-05-21, H100 80GB, 2h budget, public)

Headline target ceiling per the InferenceBench README. RTX PRO 6000 results
should not be directly compared to these numbers without an H100 rerun.

| Method               | Agg.   | Sc. A | Sc. B  | Sc. C  | Sc. D |
|----------------------|-------:|------:|-------:|-------:|------:|
| SMAC3 search, 2h     | 11.53x | 4.37x | 15.23x | 46.70x | 5.69x |
| TPE search, 2h       | 11.25x | 4.48x | 14.76x | 43.46x | 5.58x |
| Random search, 2h    | 10.20x | 4.21x | 11.34x | 41.81x | 5.42x |
| Claude Sonnet 4.6    |  8.08x | 3.47x | 12.03x | 33.93x | 3.01x |
| vLLM default         |  4.05x | 1.25x |  2.25x | 48.69x | 1.96x |
| SGLang default       |  3.92x | 1.22x |  1.77x | 51.12x | 2.14x |
| TGI default          |  3.30x | 1.14x |  1.37x | 41.94x | 1.80x |
| PyTorch baseline     |  1.00x | 1.00x |  1.00x |  1.00x | 1.00x |

## Update History

- 2026-05-24 — Initial creation. Preflight passes for all scenarios via
  imported scoring assets (`rtxpro6000-seed248`). No SENPAI-validated
  launcher yet; first round of student PRs initialized for A, B, C in
  parallel under shared-GPU slot coordination.
