# SENPAI Research State — InferenceBench

- **As of:** 2026-05-28 21:08 UTC
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
| B | 1/tpot.p50 | **4.450x** | vLLM FP8wt + n-gram spec25/12 | #179 | 29% of 15.23x |
| C | geomean req/s | **29.768x** | SGLang LPM + radix + FP8 wt + FP8 KV + mem 0.90 | #181 | 64% of 46.70x |
| D | geomean (1/ttft, 1/tpot, req/s) | **2.218x** | vLLM FP8 + n-gram spec10 | #152 | 39% of 5.69x |

## Active experiments

| Student | PR | Scenario | Hypothesis | Status |
|---|---:|---|---|---|
| fern | #187 | C | SGLang engine upgrade (newer pip release) + FlashInfer probe (arm1 triton, arm2 flashinfer) | **assigned 20:57 UTC** |
| frieren | #186 | A | vLLM 0.12 + FlashInfer unblock on SM120 (arm1 vLLM12+FA control; arm2 vLLM12+FlashInfer) | **assigned 20:35 UTC** |
| tanjiro | #185 | B | max-num-seqs=1: quick **8.21x** (vs PR #179 quick 3.856x = +113%!); arm1 full eval running | **partial result 20:59 UTC** |

## Completed experiments this session

| PR | Student | Scenario | Result | Status |
|---:|---|---|---|---|
| #184 | fern | C | cps=16384 full 29.690x (-0.26% vs PR #181) | CLOSED — chunked-prefill-size is tail-shape knob only (Rule #19); cps axis exhausted |
| #183 | frieren | A | SGLang 1.549x (-18.5% vs PR #156 1.881x) | CLOSED — SGLang Triton prefill 21% slower than vLLM FA on SM120 |
| #179 | tanjiro | B | **4.450x** FP8wt + spec25/lookup12 (+14.5% over PR #149 3.888x) | MERGED — new Sc B best; arm2 (spec20) quality_failed at full |
| #182 | fern | C | mem 0.95 full 29.727x = -0.14% vs PR #181 (noise) | CLOSED — did_not_improve; mem-fraction lever exhausted at 0.90 |
| #180 | frieren | A | tokens 16384 full 1.8658x = -0.83% vs PR #156 1.881x | CLOSED — did_not_improve; tokens beyond 8192 not a Sc A lever |
| #181 | fern | C | **29.768x** SGLang mem 0.85→0.90 on PR #172 (+0.80%) | MERGED — new Sc C best |
| #172 | fern | C | **29.532x** SGLang FP8wt+FP8KV (+7.4% over 27.497x) | MERGED — superseded by #181 |
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

14. **Rule #18 — Sc C quick→full amplification factor (REFINED by PR #182):** Sc C quick is concurrency-bound at n=4 burst. Lever-induced gains amplify ~4-9× quick→full **when the lever has real effect**. PR #181 mem-fraction 0.85→0.90: quick +0.18% → full +0.80% (~4.4×). PR #172 FP8 KV: quick +1.2% suppression → full +7.4% (effectively ~6× when sign-corrected). **However, PR #182 demonstrated that quick gaps within ±0.5% can also be pure noise that do NOT amplify** (mem 0.95: quick +0.19% vs arm1 → full -0.14%). Implication: rule #18 applies when the lever has mechanism; saturated-mechanism arms (e.g., dead-capacity beyond KV ceiling) stay at noise levels. Promotion to full eval is still informative even at +0.1-0.3% quick, but expect ~50% of such arms to be noise.

16. **Rule #19 — Sc C chunked-prefill-size is a tail-shape knob, not a headline knob (PR #184):** cps=16384 (full): -0.26% headline, but burst TPOT p99 −50% (0.65→0.32s) AND burst TTFT p90 +42% worse (1.071→1.526s). The two effects cancel in the geomean-of-3-profiles primary. Future Sc C work should target levers that change per-step decode throughput (engine version, speculation, attention kernel) — not prefill/decode scheduling rebalancing.

15. **Sc C local optimum near 30x:** 4 consecutive Sc C wins (PR #144 24.305x → PR #151 27.497x +13.1% → PR #172 29.532x +7.4% → PR #181 29.768x +0.80%). Marginal gains shrinking exponentially. Mem-fraction lever exhausted at 0.90. Next mechanisms to test: chunked-prefill-size (for burst tail), CUDA graph batch sizes, or fundamentally new approaches (spec decoding, EAGLE/MTP heads). 64% of H100 SMAC3 ceiling — closing in but real gains require new mechanism.

## Current research focus

**Priority ranking: Sc B (29% of SMAC3 ceiling) > Sc D (39%) > Sc A (42%) > Sc C (64%).**

### Active hypothesis queue (in priority order)

1. **Sc C fern #187 (assigned 20:57 UTC):** SGLang engine upgrade (latest pip release) + FlashInfer backend probe. 2-arm: arm1 newer SGLang + PR #181 config; arm2 newer SGLang + flashinfer backend. Quick ETA ~21:25 UTC. Hard-close if quick shows <+0.5% gain.

2. **Sc B tanjiro #185 (assigned 20:32 UTC):** max-num-seqs=1 on PR #179 base (phantom KV freed at conc=1). Quick ETA ~21:00 UTC; full ETA ~22:00 UTC.

3. **Sc A frieren #186 (assigned 20:35 UTC):** vLLM 0.12 + FlashInfer unblock on SM120. Quick ETA ~21:05 UTC (add ~5 min for vLLM 0.12 install); full ETA ~22:05 UTC.

### Next experiments (after current round)

1. **Sc D: EAGLE-2 draft-model speculation** — n-gram local optimum confirmed. High-risk-high-value.

2. **Sc A: vLLM 0.12+ per-PR venv** — Conditional on PR #183 outcome. vLLM 0.12 unblocks FlashInfer cascade (20%+ Sc A TTFT potential).

3. **Sc B: spec depth fine-tune with FP8** — check if spec27/28 with FP8 gives marginal gain above spec25/FP8 (small, but the SMAC3 gap at 29% is still large).

4. **Sc C burst-tail mechanism**: If chunked-prefill-size has no effect, try `--cuda-graph-bs` or `--max-running-requests` sweep.

## Operational notes

- SGLang relaunch: bundle `lib/libnuma.so.1`; per-PR venv `/tmp/inferencebench-engine-venvs/sglang-pr-<slug>` auto-bootstrapped.
- `--enable-radix-cache` does NOT exist in SGLang 0.5.x (radix ON by default; use `--disable-radix-cache` if needed).
- `TRITON_ATTN` is vLLM 0.11 backend name; FlashInfer attention blocked on SM120.
- Hard close: ~22:38 UTC (4h remaining). Room for ~2 full-eval cycles per student.
- Quality gate: MMLU-Pro tau=0.95 (ratio ≥ 0.95), n=500.
- VRAM budget: 97.9 GiB total; practical ceiling ~93 GiB.
- n=16 quality gate unreliable (binomial SE ~11.5pp). Use as raw floor only; n=500 is authoritative.
