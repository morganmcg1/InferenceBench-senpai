# SENPAI Research State — InferenceBench

- **As of:** 2026-05-28 14:50 UTC
- **Run tag / advisor branch:** `ib-20260528-12h-r2`
- **Hardware (active):** NVIDIA RTX PRO 6000 (~96 GB) — shakedown only; not
  leaderboard-comparable to the H100 reference snapshot in `program.md`.
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Students:** fern, frieren, tanjiro (1 dedicated GPU per pod).
- **W&B:** `wandb-applied-ai-team/inferencebench-senpai`
- **Most recent human research-team direction:** none on this branch yet.

## Current baselines (RTX PRO 6000)

| Scenario | Metric | Best speedup | Engine | PR | vs H100 SMAC3 ceiling |
|---|---|---:|---|---:|---:|
| A | 1/ttft.p50 | **1.866x** | vLLM FP8 weights | #137 | 42% of 4.48x |
| B | 1/tpot.p50 | **3.550x** | vLLM n-gram spec15/8 | #141 | 23% of 15.23x |
| C | geomean req/s | **24.305x** | SGLang LPM + radix | #144 | 52% of 46.70x |
| D | geomean (1/ttft, 1/tpot, req/s) | **2.073x** | vLLM FP8 + n-gram | #139 | 36% of 5.69x |

## Active experiments

| Student | PR | Scenario | Hypothesis | Status |
|---|---:|---|---|---|
| fern | #147 | D | SGLang LPM + radix cache (port of PR #144 +15.2% mechanism to Sc D) | **assigned 14:50 UTC** |
| frieren | #148 | A | FlashInfer prefill attention + FP8 weights (kernel-level TTFT attack) | **assigned 14:55 UTC** |
| tanjiro | #146 | B | FP8 weights + spec15/lookup8 composition (expected 5–7x) | **WIP** |

All 3 student GPUs occupied.

## Completed experiments this session

| PR | Student | Scenario | Result | Status |
|---:|---|---|---|---|
| #137 | frieren | A | 1.866x FP8 weights | MERGED — current Sc A best |
| #136 | fern | B | 2.687x n-gram spec5/4 | MERGED — superseded by #141 |
| #138 | tanjiro | D | 1.247x SGLang default | MERGED — superseded by #139 |
| #140 | fern | C | 21.052x vLLM seqs=64 | MERGED — superseded by #142 |
| #139 | frieren | D | 2.073x vLLM FP8+n-gram | MERGED — current Sc D best |
| #141 | tanjiro | B | 3.550x n-gram spec15/8 | MERGED — current Sc B best (+32.2%) |
| #142 | fern | C | 21.098x vLLM seqs=128 | MERGED — superseded by #144 |
| #143 | frieren | A | 1.8603x FP8+n-gram | CLOSED — spec metric-orthogonal to TTFT |
| #144 | fern | C | 24.305x SGLang LPM+radix | MERGED — current Sc C best (+15.2%) |
| #145 | frieren | A | 1.380x FP8 weights + FP8 KV cache | CLOSED — FP8 KV regresses TTFT -26% on SM120 concurrency-1 |

## Key learnings

1. **FP8 weight quantization** wins on prefill-bound workloads (Sc A 1.87x, Sc D TTFT component). On bandwidth-bound single-stream prefill, FP8 halves weight-read time proportionally.

2. **N-gram speculative decoding** wins on decode-bound workloads (Sc B 3.55x, Sc D TPOT). Depth scales superlinearly on Sc B's 8192-token outputs: spec15 at 3.55x vs spec5 at 2.69x (+32%). TPOT cut from 9.36ms → 7.08ms.

3. **FP8 + n-gram compose additively on Sc D** (2.073x = 1.83x TTFT × 2.40x TPOT) because they target independent pipeline stages. Composition is a confirmed architectural pattern.

4. **SGLang LPM scheduler unlocks radix cache on Sc C** (+15.2% over vLLM best, 24.305x). In SGLang 0.5.x, radix cache is ON by default — `--schedule-policy lpm` reorders the request queue to maximise prefix-tree hits. FCFS scheduling wastes the radix cache.

5. **FP8 hurts on Sc C** (scheduler/KV-memory-bound, not bandwidth-bound). FP8 dequant overhead dominates at high concurrency.

6. **N-gram spec is metric-orthogonal to Sc A** (TTFT-only primary metric — speculation only improves TPOT, which Sc A doesn't score). Closed PR #143 as a learning, not a failure.

7. **FP8 KV cache hurts Sc A TTFT** (PR #145, -26% regression). At concurrency 1 with an 8192-token prefill, the attention kernel is compute-bound on SM120, not KV-bandwidth-bound. FP8 KV adds dequantization overhead with zero bandwidth benefit at this operating point. The vLLM FP8 bandwidth-reduction levers are now exhausted for Sc A; next attack is the attention kernel itself (FlashInfer).

7. **Quick-to-full ratios for speculative decoding:** Sc B consistently ~51–76% (quick overestimates due to repetitive samples at n=4). Treat quick speculative results as upper bounds.

## Current research focus

**Primary focus: close the gap to H100 SMAC3 ceilings, especially Sc B (23%) and Sc D (36%).**

The SGLang LPM mechanism is now confirmed on Sc C. Three active hypotheses test whether it transfers to other scenarios and whether composition levers work on each scenario:

- **Sc D (fern #147):** SGLang LPM + radix on 4096-token inputs. These longer prefixes should hit the radix tree harder than Sc C's 1024-token inputs. Composition with FP8 and/or n-gram spec (if SGLang supports it) in later arms.

- **Sc A (frieren #145):** FP8 KV-cache to shrink attention memory footprint at prefill. FP8 KV frees KV-block VRAM, allowing a larger effective batch even at concurrency=1. Potential for another 1.2–1.5x gain on top of the 1.866x FP8-weight baseline.

- **Sc B (tanjiro #146):** FP8 weights + spec15/lookup8 composition. Mirroring the confirmed Sc D composition pattern (FP8 cuts TTFT bandwidth, spec cuts TPOT forward passes). Expected 5–7x if effects are independent, which they should be on Sc B's 1024in/8192out workload.

## Potential next experiments (after current round)

See full list in `research/RESEARCH_IDEAS_2026-05-28_14:45.md`. Top priorities ordered by expected impact:

1. **Sc D: vLLM spec15/lookup8 composition** (researcher H1) — upgrade PR #139 Sc D winner from spec5/lookup4 to spec15/lookup8 (two integer changes). Confirmed +32% on Sc B; should give ~2.4–2.6x on Sc D. LOW risk. Complements fern's #147 (SGLang engine test) — different axes.

2. **Sc B deeper speculation sweep (spec20/lookup10, spec25/lookup12)** (researcher H6) — 3.550x vs 15.23x ceiling = large gap. The spec depth sweep hasn't hit saturation yet; arm1→arm2→arm3 at (7,4)→(10,6)→(15,8) was superlinear. Plus try `prompt_lookup_min=1` as a free add-on.

3. **Sc C: SGLang FP8 KV cache + mem-fraction push** (researcher H2+H5) — 15 GiB of unused VRAM headroom in PR #144. FP8 KV halves KV block size, allowing more concurrent states. `--mem-fraction-static 0.92` + `--kv-cache-dtype fp8_e5m2`. Expected 8–15% req/s gain.

4. **Sc D SGLang + n-gram spec composition** (researcher H3) — after fern's #147 establishes SGLang LPM baseline on Sc D, compose with spec. SGLang NGRAM flag: `--speculative-algorithm NGRAM`. Note: SGLang NGRAM may disable overlap scheduler (measure empirically).

5. **Sc C: SGLang + n-gram spec** (researcher H4) — 1024-token decode per request may accept spec proposals. Add `--speculative-algorithm NGRAM --num-speculative-tokens 5/10` to PR #144 arm3 config. Expected 3–12% gain on constant/poisson profiles.

6. **Sc A: TensorRT-LLM** — if FlashInfer (PR #148) plateaus, TRT-LLM with fused FP8 kernels targets the 1.866x → 4.48x gap. High install risk; reserve for late session.

## Operational notes

- SGLang relaunch contract: bundle `lib/libnuma.so.1` in every SGLang launcher PR; per-PR venv at `/tmp/inferencebench-engine-venvs/sglang-pr-<slug>` auto-bootstrapped via `uv venv` + `pip install sglang[all]==0.5.12.post1`.
- `--enable-radix-cache` does NOT exist in SGLang 0.5.x. Use `--disable-radix-cache` if needed; radix cache is ON by default.
- H100 reference timing: ~22:38 UTC hard close for this 12-hour shakedown. With ~7.75 hours remaining, we have room for ~2–3 more full-eval cycles per student.
- Quality gate: MMLU-Pro tau=0.95 (observed/baseline ≥ 0.95), n=500. Gate floor: 0.283.
- VRAM budget: 97.9 GiB total; practical ceiling ~93 GiB with fragmentation.
