# SENPAI InferenceBench Research Results — ib-20260524-ready-r3

## 2026-05-24 08:08 — PR #37: Scenario B vLLM FP8 KV-cache, single-stream TPOT-focused burst

- Branch: `r3-frieren/scB-vllm-fp8-tpot`
- W&B run: `zm6tpgv4` (group `scB-vllm-fp8-tpot`)
- Hypothesis: vLLM + `--kv-cache-dtype fp8` + `--max-num-seqs 1` + chunked-prefill OFF + prefix-caching OFF + CUDA graphs ON + `--block-size 16` would deliver a TPOT win on burst concurrency-1 by halving KV bandwidth while preserving MMLU-Pro accuracy on Mistral-7B-Instruct-v0.3.

### Results (RTX PRO 6000 shakedown, reduced-N, not leaderboard)

| Metric | Value |
|---|---|
| `scenario/B/speedup_over_pytorch` | **1.568x** |
| Raw `1/tpot.p50` (this run) | 62.334 (8 reqs) |
| Raw `1/tpot.p50` (PyTorch baseline ref) | 39.757 (64 reqs) |
| Generation throughput | 61.96 tok/s |
| TPOT p50 / p90 / p99 | 0.01604 / 0.01848 / 0.01884 s |
| TTFT p50 / p99 | 0.0652 / 0.0680 s |
| Request throughput | 0.01261 req/s |
| Success / failure | 8 / 0 |
| VRAM peak | 90099 / 97887 MiB |
| **Quality gate `pass`** | **FALSE** (ratio 0.839 < τ 0.95) |
| MMLU-Pro observed (n=64) | 0.250 |
| MMLU-Pro baseline (n=500) | 0.298 |
| MMLU-Pro ratio | 0.839 |
| Caveats | `request_limit_8_of_64`, `mmlupro_n_64_of_500`, `quality_gate_failed_ratio_0.839_below_0.95` |
| Cold relaunch | PASS (45s cold start, `/v1/models` 200, all knobs honored) |

### Commentary & analysis

- **Speed delivered.** 1.568x is real and positive — vLLM-FP8-KV beats PyTorch baseline on Scenario B's `1/tpot.p50` raw objective.
- **But the launcher is disqualified by the quality gate.** Per the InferenceBench scoring contract, MMLU-Pro ratio must be ≥ 0.95 for a launcher to be a valid speedup_over_pytorch claim. 0.839 falls well below.
- **vLLM-default reference snapshot on H100 is ~2.25x**; this run sits at 1.568x. Hardware effect is real — RTX PRO 6000 PyTorch baseline `1/tpot.p50` = 39.757 vs H100 reference would be lower in raw tokens but the *ratio* matters. Either PyTorch is relatively stronger on RTX PRO 6000 or vLLM-FP8 is relatively weaker on Blackwell vs Hopper.
- **Statistical noise is high.** Only 8 burst samples → `tpot.p50` is the median of 8 data points. Sufficient for direction but not for fine ranking.
- **Root cause uncertainty.** Quality fail could be (a) FP8 KV-cache numerics genuinely hurt Mistral-7B-Instruct decoding on MMLU-Pro, or (b) `--quality_max_new_tokens 2048` truncation effect specific to this launcher config. Round 2 must disambiguate.
- **Disposition:** **Close** PR — not a winner per scoring contract. Round-2 follow-up: FP16 KV variant to isolate the lever.

### Decision

- Closed without merge (quality gate fail).
- BASELINE.md remains unchanged (no winner for scenario B yet).
- Launcher recipe at commit `9268663` is preserved in the closed PR for round-2 reference.

### Next variant to try

- `--kv-cache-dtype auto` (FP16 KV), same scheduler levers, same reduced-N (`--request-limit 8`, `mmlupro_n=64`). If FP16 KV passes quality, TPOT regresses a fraction but the launcher is valid. Assign as r3-frieren round-2 PR.

## 2026-05-24 08:36 — PR #38: Scenario A vLLM FP8 KV-cache, bigbatch-prefill TTFT-focused burst

- Branch: `r3-fern/scA-vllm-bigbatch-prefill`
- W&B run: `b0h44kvt` (group `scA-vllm-bigbatch-prefill`)
- Hypothesis: vLLM + `--kv-cache-dtype fp8` + `--max-num-batched-tokens 16384` + `--no-enable-chunked-prefill` + `--no-enable-prefix-caching` + CUDA graphs ON would deliver a TTFT.p50 win on burst concurrency-1 by feeding the 8k prompt as a single forward pass (no chunk scheduling overhead) while halving KV bandwidth.

### Results (RTX PRO 6000 shakedown, reduced-N, not leaderboard)

| Metric | Value |
|---|---|
| `scenario/A/speedup_over_pytorch` | **1.030x** |
| Raw `1/ttft.p50` (this run) | 2.349 (16 reqs) |
| Raw `1/ttft.p50` (PyTorch baseline ref) | 2.280 (128 reqs) |
| Generation throughput | 56.47 tok/s |
| TTFT p50 / p90 / p99 | 0.4257 / 0.4793 / 0.4937 s |
| Baseline TTFT p50 / p99 | 0.4385 / 0.9756 s |
| TTFT p99 tail speedup | **1.98x** |
| TPOT p50 / p90 / p99 | 0.01686 / 0.02381 / 0.03177 s |
| Request throughput | 0.0922 req/s |
| Success / failure | 16 / 0 |
| VRAM peak | 91217 MiB |
| **Quality gate `pass`** | **FALSE** (ratio 0.839 < τ 0.95) |
| MMLU-Pro observed (n=64) | 0.250 |
| MMLU-Pro baseline (n=500) | 0.298 |
| MMLU-Pro ratio | 0.839 |
| Caveats | `request_limit_16_of_128`, `mmlupro_n_64_of_500`, `quality_gate_failed_ratio_0.839_below_0.95`, `kv_cache_dtype_fp8_regresses_mmlu_pro_on_mistral_7b` |
| Cold relaunch | PASS (17s cold start, `/v1/models` 200, all knobs honored) |

### Commentary & analysis

- **Small but real speed win.** 1.030x on median TTFT — below the H100 vLLM-default reference of 1.25x. The `--max-num-batched-tokens 16384` lever was likely a near-no-op given chunked prefill was already disabled (vLLM feeds the 8k prompt as one forward pass regardless once `--no-enable-chunked-prefill`).
- **Tail-latency win is interesting.** TTFT p99 of 0.494s vs baseline 0.976s = 1.98x tail speedup. The shape of bigbatch-prefill is correct; the median is bottlenecked elsewhere (prefill matmul + attention kernel cost, not scheduling).
- **Quality gate fail = launcher disqualified.** Per scoring contract, MMLU-Pro ratio must be ≥ 0.95 for a valid speedup_over_pytorch claim. 0.839 falls far below.
- **CRITICAL FINDING — FP8 KV is a systematic Mistral-7B quality killer.** PR #37 (scenario B) and PR #38 (scenario A) produced **identical** MMLU-Pro 0.250 vs 0.298 baseline, ratio 0.839 to 3 decimals. Two independent scenarios, same exact 16/64 correct on quality eval — deterministic, not noise. This rules out per-scenario artifacts and points squarely at `--kv-cache-dtype fp8` numerics for this model/engine/hardware combination.
- **Statistical noise on speed is high.** 16 burst samples → `ttft.p50` median of 16 data points. Sufficient for direction.
- **Disposition:** **Close** PR — quality gate disqualifies the launcher.
- **Bug report shipped on advisor branch.** r3-fern flagged a real footgun in `senpai/summarize_metrics.py`: PyTorch baseline JSONs wrap profiles under `baseline.profiles.burst`, but `primary_metric` reads `metrics["profiles"]["burst"]` directly, forcing `--baseline-primary <float>` as workaround. Advisor commit `3b34a4a` unwraps the `baseline` envelope before calling `primary_metric`. Future students unblocked.

### Decision

- Closed without merge (quality gate fail).
- BASELINE.md remains unchanged (no winner for scenario A yet).
- Launcher recipe at commit `023f5b0` is preserved in the closed PR for round-2 reference.

### Next variant to try

- `--kv-cache-dtype auto` (FP16 KV), same scheduler levers, same reduced-N. Mirror structure of PR #50 (r3-frieren round 2 scenario B FP16 variant). Assign as r3-fern round-2 PR with slug `vllm-fp16-bigbatch-prefill`.
