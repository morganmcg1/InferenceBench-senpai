# SENPAI Research State — InferenceBench

- **As of:** 2026-05-28 19:18 UTC
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
| A | 1/ttft.p50 | **1.881x** | vLLM FP8 weights + max-num-seqs=1 | #156 | 42% of 4.48x |
| B | 1/tpot.p50 | **3.888x** | vLLM n-gram spec25/12 | #149 | 26% of 15.23x |
| C | geomean req/s | **29.532x** | SGLang LPM + radix + FP8 wt + FP8 KV | #172 | 63% of 46.70x |
| D | geomean (1/ttft, 1/tpot, req/s) | **2.218x** | vLLM FP8 + n-gram spec10 | #152 | 39% of 5.69x |

## Active experiments

| Student | PR | Scenario | Hypothesis | Status |
|---|---:|---|---|---|
| fern | #181 | C | mem-fraction push on PR #172 winner (FP8wt+FP8KV, 0.90 and 0.95) | **assigned 19:18 UTC** |
| frieren | #180 | A | `--max-num-batched-tokens` sweep beyond 8192 (3-arm: 16384, 32768, 65536) on PR #156 base | **assigned 19:10 UTC** |
| tanjiro | #179 | B | FP8 weights composition: arm2 fp8+spec20 quick 7.789x → full eval running (rule #17 caution: Sc B quick→full may collapse) | **arm2 full eval running** |

All 3 student GPUs occupied.

## Completed experiments this session

| PR | Student | Scenario | Result | Status |
|---:|---|---|---|---|
| #172 | fern | C | **29.532x** SGLang FP8wt+FP8KV (+7.4% over 27.497x) | MERGED — new Sc C best |
| #166 | frieren | B | spec25/min=1: quick 5.601x (+44%), full **3.329x (-14.4%)** | CLOSED — quick→full collapse; rule #17 established |
| #156 | fern | A | **1.881x** FP8+seqs=1+tok=8192 (+0.78% over 1.866x) | MERGED — new Sc A best |
| #164 | tanjiro | D | FP8 KV arm1/arm2 -10%, arm3 BF16 control tied 2.208x | CLOSED — did_not_improve; rule #16 established |
| #173 | tanjiro | D | Prefill scheduling sweep: all arms tied within 0.15%, +0.5-0.6% over PR #152 quick | CLOSED — did_not_improve; Sc D scheduling exhausted |
| #153 | frieren | C | SGLang FP8 wt + FP8 KV stuck 85+ min | CLOSED — abandoned; retry assigned fern #172 |
| #149 | tanjiro | B | 3.888x spec25/lookup12 | MERGED — current Sc B best |
| #152 | fern | D | 2.218x spec10/lookup6 | MERGED — current Sc D best |
| #154 | tanjiro | D | spec11/12 fail screen | CLOSED — n-gram exhausted on Sc D |
| #151 | frieren | C | 27.497x SGLang FP8 KV | MERGED — current Sc C best |
| #147 | fern | D | 1.568x SGLang LPM+radix | CLOSED — SGLang -24.4% vs vLLM on Sc D |
| #148 | frieren | A | Triton fallback −20% | CLOSED — FlashInfer blocked 5-layer cascade |
| #145 | frieren | A | FP8 weights + FP8 KV -26% | CLOSED — FP8 KV hurts conc=1 compute-bound |
| #143 | frieren | A | FP8+n-gram metric-orthogonal on Sc A | CLOSED — spec improves TPOT, Sc A scores TTFT |
| #146 | tanjiro | B | FP8+spec15 quality fail 0.913 | CLOSED — FP8+deep spec fails quality gate |
| #144 | fern | C | 24.305x SGLang LPM+radix | MERGED — superseded by #151 |

## Key learnings

1. **FP8 weight quantization** wins on prefill-bound workloads (Sc A 1.87x, Sc D TTFT). FP8 halves weight-read bandwidth; proportional TTFT win at conc=1 bandwidth-bound prefill.

2. **N-gram speculative decoding** wins on decode-bound workloads (Sc B 3.55x, Sc D TPOT). Depth scales superlinearly on Sc B's 8192-token outputs. Sc D caps at spec10 (quality cliff).

3. **FP8 + n-gram compose additively on Sc D** (2.073x, +66%) because they target independent pipeline stages (prefill bandwidth vs decode step count).

4. **SGLang LPM scheduler unlocks radix cache on Sc C** (+15.2% → 24.305x). LPM reorders to maximise prefix-tree hits; radix cache ON by default in SGLang 0.5.x.

5. **SGLang LPM+radix is Sc C-specific** — does NOT transfer to Sc D (4-concurrency, no shared prefixes). Sc D has unique requests; LPM/radix adds overhead.

6. **FP8 KV regime threshold (rule #16):** HURTS at conc=1 (Sc A, -26%) and conc=4 (Sc D, -10%). WINS at conc=256 (Sc C, +13.1%). Both e5m2 and e4m3 give identical regressions on Sc D. Bandwidth-vs-dequant crossover sits between 4 and 256 concurrent requests. FP8 KV is **Sc C-specific** on this workload.

7. **FlashInfer blocked on vLLM 0.11 + SM120** (PR #148) — 5-layer compatibility cascade. vLLM ≥ 0.12 would unblock.

8. **FP8 + spec ≥ 15 fails quality on Sc B** (PR #146, 0.913 ratio). Rule: FP8 + deep spec (depth ≥ 15) interacts with quality gate on Sc B.

9. **Sc D scheduling lever exhausted (rule from PR #173):** PR #152's chunked+4096 is the local optimum. Chunked vs one-shot: +0.13% (wash). Token budget 4096→16384: +0.62% (below threshold). TTFT/TPOT at hardware limit.

10. **Sc D n-gram local optimum is spec10/lookup6/min=2** (PR #152). Quality cliff at spec≥11. min=1 gains +4.7% speed but drops quality to 0.926 < gate. All single-flag levers on PR #152 confirmed exhausted.

11. **Sc B spec depth: spec25 = peak.** spec5→15 was +32%; spec15→25 only +9.5%; spec30 saturates. Depth ceiling is output-length-dependent (8192-out Sc B tolerates 25; 2048-out Sc D caps at 10).

12. **Sc A is TTFT-scored only.** N-gram spec improves TPOT -27% but is metric-invisible. PR #156 took Sc A to 1.881x with FP8+seqs=1+tok=8192.

13. **Rule #17 — Sc B quick→full mapping UNRELIABLE for n-gram lookup behavior changes** (PR #166: quick 5.601x → full 3.329x, -41% collapse). With `lookup_min=1`, every history token seeds a 25-token speculative branch; at n=4 burst rare lucky matches dominate, at n=128 burst rejected branches dominate. Quality DID absorb the precision loss (ratio 1.054) — only speed mechanism failed. Spec depth changes (PR #149) remain stable quick→full; only lookup_min/max changes shift acceptance rate distributions.

## Current research focus

**Priority ranking: Sc B (26% of SMAC3 ceiling) > Sc D (39%) > Sc A (42%) > Sc C (59%).**

### Active hypothesis queue (in priority order)

1. **Sc B tanjiro #179 (HIGH PRIORITY — arm2 full eval running ETA ~20:00 UTC):** FP8 + spec20/lookup10 quick = **7.789x** (+100% over PR #149!). Rule #17 caution: Sc B quick→full unreliable for n-gram dynamics changes — arm2 cuts spec depth 25→20 AND adds FP8. Quality at n=16 floor (3/16). Full eval will decide. Fallback: arm1 (fp8+spec25 = 3.856x ≈ PR #149, no improvement if arm2 collapses).

2. **Sc A frieren #180 (assigned 19:10 UTC):** `--max-num-batched-tokens` sweep beyond 8192 (16384, 32768, 65536). Single-flag tuning, no quality risk.

3. **Sc C fern #181 (NEW — assigned 19:18 UTC):** mem-fraction push on PR #172 winner (FP8wt+FP8KV). Test 0.90 and 0.95 to confirm whether higher mem allocation improves throughput at 256-conc. PR #172 arm2 (mem 0.92) was flat at quick (concurrency-bound) — full eval will discriminate.

### Next experiments (after current round)

1. **Sc D: EAGLE-2 draft-model speculation** — n-gram local optimum confirmed. High-risk-high-value. Priority after current round closes.

2. **Sc A: vLLM 0.12+ per-PR venv** — Conditional on PR #180 outcome. vLLM 0.12 unblocks FlashInfer cascade (20%+ Sc A TTFT potential).

3. **Sc B compound**: If tanjiro #179 arm2 holds at full eval (FP8 + spec20), next Sc B experiment: compose arm2 winner with other levers (e.g. higher mem-fraction, different spec config).

## Operational notes

- SGLang relaunch: bundle `lib/libnuma.so.1`; per-PR venv `/tmp/inferencebench-engine-venvs/sglang-pr-<slug>` auto-bootstrapped.
- `--enable-radix-cache` does NOT exist in SGLang 0.5.x (radix ON by default; use `--disable-radix-cache` if needed).
- `TRITON_ATTN` is vLLM 0.11 backend name; FlashInfer attention blocked on SM120.
- Hard close: ~22:38 UTC (4h remaining). Room for ~2 full-eval cycles per student.
- Quality gate: MMLU-Pro tau=0.95 (ratio ≥ 0.95), n=500.
- VRAM budget: 97.9 GiB total; practical ceiling ~93 GiB.
- n=16 quality gate unreliable (binomial SE ~11.5pp). Use as raw floor only; n=500 is authoritative.
