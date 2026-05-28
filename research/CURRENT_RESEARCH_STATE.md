# SENPAI Research State — InferenceBench

- **As of:** 2026-05-28 16:35 UTC
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
| C | geomean req/s | **27.497x** | SGLang LPM + radix + FP8 KV | #151 | 59% of 46.70x |
| D | geomean (1/ttft, 1/tpot, req/s) | **2.073x** | vLLM FP8 + n-gram | #139 | 36% of 5.69x |

## Active experiments

| Student | PR | Scenario | Hypothesis | Status |
|---|---:|---|---|---|
| fern | #152 | D | vLLM spec depth upgrade: PR #139 spec5/lookup4 → spec15/lookup8 (researcher H1) | Quick probes done 16:01 UTC; **spec15 fails quality regardless of dtype**, arm2 (spec10) in full eval |
| frieren | #153 | C | SGLang FP8 weights + FP8 KV composition (extend PR #151) — ablation arm3 isolates weight contribution | **assigned 16:25 UTC** |
| tanjiro | #149 | B | Deeper n-gram spec sweep: spec20/25/30, BF16 (no FP8), extend PR #141 | **arm2 spec25/lookup12 = 3.888x** (+9.5%, quality 1.013) — sent back 16:35 UTC for BASELINE.md rebase |

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
| #147 | fern | D | 1.568x SGLang LPM+radix+FP8+NGRAM | CLOSED — did_not_improve, -24.4% vs PR #139; LPM/radix doesn't transfer (no shared prefixes), SGLang NGRAM weaker than vLLM prompt-lookup |
| #148 | frieren | A | 1.484x Triton fallback (did_not_improve) | CLOSED — FlashInfer blocked on vLLM 0.11/FlashInfer 0.6 stack (5-layer cascade) |
| #149 | tanjiro | B | spec20/25/30 BF16 — quick probes pending | **assigned 15:15 UTC** |
| #150 | frieren | C | SGLang FP8 KV + mem 0.92 — auto-merged by GH due to branch collision | CLOSED — reissued as #151 |
| #151 | frieren | C | 27.497x SGLang FP8 KV (fp8_e5m2) — MERGED new Sc C best +13.1% |
| #149 | tanjiro | B | **3.888x arm2 spec25/lookup12 (+9.5%)** — review-ready, sent back 16:35 UTC for rebase after PR #151 BASELINE.md conflict |
| #152 | fern | D | spec15 fails quality on Sc D regardless of dtype (n=16 screen); arm2 spec10 in full eval (quick 2.033x ≈ PR #139 baseline) |
| #153 | frieren | C | SGLang FP8 weights + FP8 KV composition | **assigned 16:25 UTC** |

## Key learnings

1. **FP8 weight quantization** wins on prefill-bound workloads (Sc A 1.87x, Sc D TTFT component). On bandwidth-bound single-stream prefill, FP8 halves weight-read time proportionally.

2. **N-gram speculative decoding** wins on decode-bound workloads (Sc B 3.55x, Sc D TPOT). Depth scales superlinearly on Sc B's 8192-token outputs: spec15 at 3.55x vs spec5 at 2.69x (+32%). TPOT cut from 9.36ms → 7.08ms.

3. **FP8 + n-gram compose additively on Sc D** (2.073x = 1.83x TTFT × 2.40x TPOT) because they target independent pipeline stages. Composition is a confirmed architectural pattern.

4. **SGLang LPM scheduler unlocks radix cache on Sc C** (+15.2% over vLLM best, 24.305x). In SGLang 0.5.x, radix cache is ON by default — `--schedule-policy lpm` reorders the request queue to maximise prefix-tree hits. FCFS scheduling wastes the radix cache.

5. **FP8 hurts on Sc C in vLLM** (scheduler/KV-memory-bound, not bandwidth-bound). FP8 dequant overhead dominates at high concurrency. **SGLang FP8 KV is a different mechanism** — being tested in PR #150 (different regime: KV block size reduction, not weight dequant).

6. **N-gram spec is metric-orthogonal to Sc A** (TTFT-only primary metric — speculation only improves TPOT, which Sc A doesn't score). Closed PR #143 as a learning, not a failure.

7. **FP8 KV cache hurts Sc A TTFT** (PR #145, -26%). At concurrency 1 with an 8192-token prefill, the attention kernel is compute-bound on SM120, not KV-bandwidth-bound. FP8 KV adds dequantization overhead with zero bandwidth benefit at this operating point. This is regime-specific — Sc C's 256-concurrency, KV-memory-bound workload is expected to behave differently.

8. **FP8 + spec15 composition fails quality gate on Sc B** (PR #146, 0.913 ratio). +7.9% speed but -8.7% quality vs PR #141 baseline. FP8 weight precision drift is amplified by greedy speculative verify at depth=15. Rule: FP8 + deep spec (depth ≥ 15) fails quality on Sc B.

9. **SGLang LPM + radix does NOT transfer to Sc D** (PR #147 final, 1.568x = -24.4% vs PR #139). Sc D has 4-concurrency with unique requests — no prefix sharing, so radix cache adds overhead with no benefit. Three independent failure modes: (a) radix cache requires cross-request prefix overlap (absent in Sc D), (b) SGLang NGRAM's BFS branching (max_bfs_breadth=10) wastes compute on non-repetitive outputs — accept_len 1.5-2.0 vs vLLM's effective ~2.5, (c) SGLang NGRAM forces `disable_overlap_schedule=True` regressing TTFT. Rule: SGLang LPM+radix is **Sc C-specific** (high-concurrency, bursty, cross-request prefix structure). Engine choice matters per scenario: SGLang wins Sc C (bursty, schedulable), vLLM wins Sc D (low-concurrency, deeper-spec-decode-friendly).

10. **FlashInfer attention is blocked on vLLM 0.11 + FlashInfer 0.6.x stack** (PR #148). Five-layer SM120 incompatibility cascade (plan() arg drift, cudagraph fast-path mismatch, FP8 GEMM linker failure). Triton fallback regressed -20% vs FA2 baseline. vLLM ≥ 0.12 would unblock this. All vLLM 0.11 Sc A levers are now exhausted.

11. **FP8 KV cache wins at high concurrency** (PR #151, +13.1% on Sc C). Halves KV memory footprint → doubles in-flight KV token capacity (545k → 1,090k). Regime-specific: wins at Sc C (256 concurrent requests, KV-bandwidth-bound), regresses at Sc A (conc=1, compute-bound). `fp8_e5m2` accepted by SGLang 0.5.12.post1 on SM120 Blackwell.

12. **Deep spec depth quality risk is dtype-independent on Sc D** (PR #152 quick probes, n=16 only — pending full confirmation). Both FP8+spec15 and BF16+spec15 produce quality_ratio 0.629 on Sc D, vs spec10 passing at 1.049. This is at odds with Sc B where PR #149 confirmed BF16+spec25 passes at full quality (n=500, ratio 1.013). Hypothesis: Sc D's 2048-token outputs have less repetition for n-gram matching → greedy spec verify at high depth produces more error-amplifying mispredictions per output token. Sc D spec sweet spot is shallower than Sc B's. Awaiting arm2 (spec10) full eval to confirm spec10 is a viable Sc D depth.

13. **Sc B spec depth has strong diminishing returns past 15** (PR #149, 3.888x at spec25 = +9.5% over PR #141's 3.550x at spec15). The spec5→spec15 jump was +32% on Sc B; spec15→spec25 is only +9.5%, and spec30 saturates (4.30x quick vs spec25's 8.64x quick — verify-step compute overhead exceeds gains). Sc B spec depth peak appears to be at spec25/lookup12.

## Current research focus

**Primary focus: close the gaps to H100 SMAC3 ceilings, especially Sc B (23%) and Sc D (36%).**

### Active hypothesis queue (in priority order)

1. **Sc B tanjiro #149:** Deeper spec sweep (spec20/25/30 BF16). RESULT: **arm2 spec25/lookup12 = 3.888x (+9.5%)**, quality 1.013, 64/64 successes. Sent back for rebase 16:35 UTC due to PR #151 BASELINE.md conflict. spec30 saturates (verify-compute overhead) — spec25 is the new Sc B sweet spot.

2. **Sc C frieren #153:** SGLang FP8 weights composed with FP8 KV (extend PR #151 winner). PR #151 confirmed FP8 KV +13.1% (24.305x→27.497x). Testing whether SGLang FP8 weight quantization adds further weight-bandwidth reduction. 3-arm ablation: arm1 FP8wt+FP8KV, arm2 arm1+mem0.92, arm3 FP8wt only (no FP8KV — isolates weight contribution). Key question: does vLLM-style "FP8 weights hurt at high concurrency" apply to SGLang's Triton kernels?

3. **Sc D fern #152:** vLLM spec depth upgrade (researcher H1). Extend PR #139 winner (FP8 + spec5/lookup4) to spec15/lookup8 — two integer changes. PR #141 showed spec5→spec15 gave +32% on Sc B. Expected +15-20% on Sc D → ~2.4x. 3-arm sweep: FP8+spec15 (full upgrade), FP8+spec10 (intermediate), BF16+spec15 (quality control). Quality risk lower than Sc B (4× less verify-step exposure with 2048 vs 8192 output tokens).

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
