# SENPAI Research State — InferenceBench

- **As of:** 2026-05-28 15:35 UTC
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
| fern | #147 | D | SGLang LPM + radix cache (port of PR #144 +15.2% mechanism to Sc D) | **quick complete, full eval running** |
| frieren | #150 | C | SGLang FP8 KV cache (`fp8_e5m2`) + mem-fraction push (0.85→0.92) — extend PR #144 | **assigned 15:35 UTC** |
| tanjiro | #149 | B | Deeper n-gram spec sweep: spec20/25/30, BF16 (no FP8), extend PR #141 | **assigned 15:15 UTC** |

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
| #145 | frieren | A | 1.380x FP8 weights + FP8 KV cache | CLOSED — FP8 KV regresses Sc A TTFT -26% at conc=1 compute-bound prefill |
| #146 | tanjiro | B | 3.83x FP8 weights + spec15/lookup8 | CLOSED — quality gate fails (0.913); FP8 drift amplified by greedy spec verify |
| #147 | fern | D | partial: quick arm4 1.506x (FP8+NGRAM) | **full eval running** — below PR #139 2.073x expected |
| #148 | frieren | A | 1.484x Triton fallback (did_not_improve) | CLOSED — FlashInfer blocked on vLLM 0.11/FlashInfer 0.6 stack (5-layer cascade) |
| #149 | tanjiro | B | spec20/25/30 BF16 — quick probes pending | **assigned 15:15 UTC** |
| #150 | frieren | C | SGLang FP8 KV + mem 0.92 — quick probes pending | **assigned 15:35 UTC** |

## Key learnings

1. **FP8 weight quantization** wins on prefill-bound workloads (Sc A 1.87x, Sc D TTFT component). On bandwidth-bound single-stream prefill, FP8 halves weight-read time proportionally.

2. **N-gram speculative decoding** wins on decode-bound workloads (Sc B 3.55x, Sc D TPOT). Depth scales superlinearly on Sc B's 8192-token outputs: spec15 at 3.55x vs spec5 at 2.69x (+32%). TPOT cut from 9.36ms → 7.08ms.

3. **FP8 + n-gram compose additively on Sc D** (2.073x = 1.83x TTFT × 2.40x TPOT) because they target independent pipeline stages. Composition is a confirmed architectural pattern.

4. **SGLang LPM scheduler unlocks radix cache on Sc C** (+15.2% over vLLM best, 24.305x). In SGLang 0.5.x, radix cache is ON by default — `--schedule-policy lpm` reorders the request queue to maximise prefix-tree hits. FCFS scheduling wastes the radix cache.

5. **FP8 hurts on Sc C in vLLM** (scheduler/KV-memory-bound, not bandwidth-bound). FP8 dequant overhead dominates at high concurrency. **SGLang FP8 KV is a different mechanism** — being tested in PR #150 (different regime: KV block size reduction, not weight dequant).

6. **N-gram spec is metric-orthogonal to Sc A** (TTFT-only primary metric — speculation only improves TPOT, which Sc A doesn't score). Closed PR #143 as a learning, not a failure.

7. **FP8 KV cache hurts Sc A TTFT** (PR #145, -26%). At concurrency 1 with an 8192-token prefill, the attention kernel is compute-bound on SM120, not KV-bandwidth-bound. FP8 KV adds dequantization overhead with zero bandwidth benefit at this operating point. This is regime-specific — Sc C's 256-concurrency, KV-memory-bound workload is expected to behave differently.

8. **FP8 + spec15 composition fails quality gate on Sc B** (PR #146, 0.913 ratio). +7.9% speed but -8.7% quality vs PR #141 baseline. FP8 weight precision drift is amplified by greedy speculative verify at depth=15. Rule: FP8 + deep spec (depth ≥ 15) fails quality on Sc B.

9. **SGLang LPM + radix does NOT transfer to Sc D** (PR #147, partial). Sc D has 4-concurrency with unique requests — no prefix sharing, so radix cache adds overhead with no benefit. arm1 (LPM+radix) gave only 1.121x quick. Best single lever was FP8 (arm2, 1.316x quick), composition arm4 (FP8+NGRAM) gave 1.506x quick but projected ~1.72x full — below PR #139 2.073x. Full eval running to confirm.

10. **FlashInfer attention is blocked on vLLM 0.11 + FlashInfer 0.6.x stack** (PR #148). Five-layer SM120 incompatibility cascade (plan() arg drift, cudagraph fast-path mismatch, FP8 GEMM linker failure). Triton fallback regressed -20% vs FA2 baseline. vLLM ≥ 0.12 would unblock this. All vLLM 0.11 Sc A levers are now exhausted.

## Current research focus

**Primary focus: close the gaps to H100 SMAC3 ceilings, especially Sc B (23%) and Sc D (36%).**

### Active hypothesis queue (in priority order)

1. **Sc B tanjiro #149:** Deeper spec sweep (spec20/25/30 BF16). The superlinear depth scaling from PR #141 (spec5→spec15: +32%) hasn't hit saturation. If spec20-30 continues the trend, Sc B could reach 5-7x. Low risk. Most important experiment in flight.

2. **Sc C frieren #150:** SGLang FP8 KV (`fp8_e5m2`) + mem-fraction push (0.85→0.92). PR #144 left 15 GiB unused VRAM. FP8 KV halves KV block size → more concurrent states at high concurrency. Different regime from Sc A's FP8 KV regression. Expected +8-15% → ~26-28x.

3. **Sc D fern #147:** SGLang LPM + radix transfer to Sc D. Quick probes show this likely underperforms PR #139 (radix-cache mechanism requires shared prefixes across requests; Sc D has 4 unique requests with 4096-token inputs). Full eval running to confirm. If did_not_improve, next Sc D attack is vLLM spec15 upgrade from PR #139's spec5.

### Next experiments (after current round)

1. **Sc D: vLLM spec15/lookup8 upgrade** — PR #139 Sc D winner uses spec5/lookup4. Upgrading to spec15/lookup8 (two integer changes) gave +32% on Sc B. Should give ~+15% on Sc D → ~2.4x. LOW risk.

2. **Sc B: `prompt_lookup_min=1`** — Quick add-on to tanjiro's depth sweep (PR #149 arm3). Single-token lookback allows matching shorter n-grams, may increase accept rate at depth=20+.

3. **Sc A: INT4 weight quantization via Marlin** — 4x bandwidth reduction vs BF16 (vs FP8's 2x). Requires pre-quantized AWQ/GPTQ checkpoint of Mistral-7B-Instruct-v0.3. Higher quality risk but could push 1.866x toward 2.5x.

4. **Sc A: vLLM 0.12+ per-PR venv** — Unblocks FlashInfer attention (layers 2-3 of the PR #148 cascade fixed in vLLM 0.12+). Same install approach as SGLang's per-PR venv.

5. **Sc D: Sc C SGLang FP8 KV transfer** — If frieren's PR #150 wins on Sc C, apply the same FP8 KV + mem-fraction mechanism to SGLang on Sc D.

## Operational notes

- SGLang relaunch contract: bundle `lib/libnuma.so.1` in every SGLang launcher PR; per-PR venv at `/tmp/inferencebench-engine-venvs/sglang-pr-<slug>` auto-bootstrapped via `uv venv` + `pip install sglang[all]==0.5.12.post1`.
- `--enable-radix-cache` does NOT exist in SGLang 0.5.x. Use `--disable-radix-cache` if needed; radix cache is ON by default.
- `TRITON_ATTN` is the correct backend name in vLLM 0.11 (not `TRITON_ATTN_VLLM_V1`).
- H100 reference timing: ~22:38 UTC hard close for this 12-hour shakedown. With ~7 hours remaining, we have room for ~2 more full-eval cycles per student.
- Quality gate: MMLU-Pro tau=0.95 (observed/baseline ≥ 0.95), n=500. Gate floor: 0.283.
- VRAM budget: 97.9 GiB total; practical ceiling ~93 GiB with fragmentation.
- Rule: FP8 + spec ≥ 15 fails quality on Sc B.
- Rule: FlashInfer attention is blocked on vLLM 0.11 + FlashInfer 0.6.x on SM120.
- Rule: FP8 KV regresses Sc A TTFT at conc=1 (compute-bound prefill). Different for Sc C (high-concurrency KV-bound) — PR #150 testing.
