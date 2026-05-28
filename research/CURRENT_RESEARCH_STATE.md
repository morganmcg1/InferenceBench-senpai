# SENPAI Research State — InferenceBench

- **As of:** 2026-05-28 18:35 UTC
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
| C | geomean req/s | **27.497x** | SGLang LPM + radix + FP8 KV | #151 | 59% of 46.70x |
| D | geomean (1/ttft, 1/tpot, req/s) | **2.218x** | vLLM FP8 + n-gram spec10 | #152 | 39% of 5.69x |

## Active experiments

| Student | PR | Scenario | Hypothesis | Status |
|---|---:|---|---|---|
| fern | #172 | C | SGLang FP8 weights composition on PR #151 winner (3-arm ablation: fp8wt+fp8kv mem0.85, +mem0.92, fp8wt-only BF16KV control) | **quick probes done 18:18 UTC; arm1 full eval running** |
| frieren | #166 | B | n-gram `prompt_lookup_min=1` sweep — arm1 quick 5.601x (+44%), arm1 full eval running | **full eval in progress since 18:18 UTC, ETA ~19:18 UTC** |
| tanjiro | #179 | B | FP8 weights composition on PR #149 spec25 winner (3-arm: fp8+spec25, fp8+spec20, fp8+spec30) | **assigned 18:35 UTC** |

All 3 student GPUs occupied.

## Completed experiments this session

| PR | Student | Scenario | Result | Status |
|---:|---|---|---|---|
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

12. **Sc A is TTFT-scored only.** N-gram spec improves TPOT -27% but is metric-invisible. All vLLM 0.11 Sc A levers exhausted.

## Current research focus

**Priority ranking: Sc B (26% of SMAC3 ceiling, biggest gap) > Sc D (39%) > Sc A (42%) > Sc C (59%).**

### Active hypothesis queue (in priority order)

1. **Sc B frieren #166 (CRITICAL — arm1 full eval ETA 19:18 UTC):** spec25/lookup12/min=1 quick = **5.601x = +44%** over PR #149. If n=500 quality ≥ 0.95: new Sc B winner at ~5.6x. If fails: falsifies rule #15's output-length insulation implication; fallback arm2 (spec25/lookup15/min=2 = 4.187x = +7.7%).

2. **Sc B tanjiro #179 (NEW — assigned 18:35 UTC):** FP8 weights composition on PR #149 spec25 winner. 3-arm: arm1 fp8+spec25 (direct composition), arm2 fp8+spec20 (quality margin), arm3 fp8+spec30 (aggressive). Key test: does FP8 × n-gram compose multiplicatively on Sc B as on Sc D? Untested combination.

3. **Sc C fern #172 (full eval running):** SGLang FP8 weights + FP8 KV composition. Quick arm1=3.991x, arm3 (FP8wt-only BF16KV control) 4.040x — full eval needed at 256-conc to see real FP8 KV benefit. Expected: FP8 weights + FP8 KV at high concurrency composes (both bandwidth reductions).

### Next experiments (after current round)

1. **Sc A: vLLM 0.12+ per-PR venv** — All vLLM 0.11 levers exhausted. vLLM 0.12 fixes the FlashInfer cascade (potential 20%+ TTFT gain).

2. **Sc D: EAGLE-2 draft-model speculation** — n-gram local optimum confirmed. Draft model (if Mistral-compatible checkpoint exists) could push beyond spec10 quality cliff. High-risk high-value.

3. **Sc B: compound with PR #166 + PR #179** — If both min=1 and FP8 win independently, compose them on the merged winner. Could stack to 5.6x × 1.1x ≈ 6.2x.

## Operational notes

- SGLang relaunch: bundle `lib/libnuma.so.1`; per-PR venv `/tmp/inferencebench-engine-venvs/sglang-pr-<slug>` auto-bootstrapped.
- `--enable-radix-cache` does NOT exist in SGLang 0.5.x (radix ON by default; use `--disable-radix-cache` if needed).
- `TRITON_ATTN` is vLLM 0.11 backend name; FlashInfer attention blocked on SM120.
- Hard close: ~22:38 UTC (4h remaining). Room for ~2 full-eval cycles per student.
- Quality gate: MMLU-Pro tau=0.95 (ratio ≥ 0.95), n=500.
- VRAM budget: 97.9 GiB total; practical ceiling ~93 GiB.
- n=16 quality gate unreliable (binomial SE ~11.5pp). Use as raw floor only; n=500 is authoritative.
