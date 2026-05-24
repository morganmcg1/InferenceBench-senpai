# SENPAI Research State

- **Date/time:** 2026-05-24 (first advisor invocation for ib-20260524-hardened-r2)
- **Human research team directives:** None received yet. Check GitHub Issues regularly.

## Current Research Focus

First round of serving optimization for InferenceBench on RTX PRO 6000 Blackwell
(~96 GB VRAM), 1 GPU shared by 3 students, 2-hour total program budget. The
starting point is vLLM defaults. We have measured PyTorch baselines for all 4
scenarios (A/B/C/D). We have NOT yet measured vLLM defaults on this hardware.

**Key context:**
- Base model: `mistralai/Mistral-7B-Instruct-v0.3`
- Hardware: 1× RTX PRO 6000 (shakedown, NOT leaderboard-comparable)
- Budget: ~2 hours total including this advisor invocation
- Public H100 references: SMAC3 15.23× (Sc B), 4.37× (Sc A), 46.70× (Sc C), 5.69× (Sc D) vs PyTorch

**Current assignments (all round 1):**
| Student | PR | Scenario | Hypothesis | GPU slot order |
|---|---|---|---|---|
| r2-frieren | #54 | B (output-heavy) | n-gram speculative decoding + FP8 KV cache | 1st |
| r2-fern | #58 | A (input-heavy) | chunked prefill + FlashInfer backend | 2nd |
| r2-tanjiro | #64 | D (balanced) | FP8 KV + chunked prefill + FlashInfer | 3rd |

Scenario C (high-load throughput) is intentionally deferred: public reference
shows vLLM default already achieves 48.69× vs PyTorch (even exceeding SMAC3
46.70×). We will investigate C only if a GPU slot opens after D, or if the
students' B/A/D runs finish faster than expected.

## Active Research Themes

1. **Decode acceleration (Scenario B):** n-gram speculative decoding is the
   most promising single lever for single-concurrency output-heavy workloads.
   If it lands 5-10×, Scenario B becomes a strong anchor result.

2. **Prefill efficiency (Scenario A):** chunked prefill + FlashInfer is the
   primary axis. The 8K input means the full prefill kernel is the bottleneck.
   FlashInfer's paged attention is especially tuned for this.

3. **Balanced serving (Scenario D):** FP8 KV + FlashInfer + chunked prefill
   bundle. Scenario D is the aggregate leaderboard signal — winning here with a
   clean recipe that works across scenarios is high-value.

4. **GPU slot coordination:** Critical constraint this run. Students must
   serialize heavy eval through `gpu_slot.py`. Idle-GPU waste must be minimized
   by having students prepare launchers and workspaces BEFORE waiting for the
   slot.

## Next Research Directions (after round 1 results)

1. **Scenario C throughput tuning:** if time allows after D, try high
   concurrency settings (`--max-num-seqs 256`, FP8 KV, prefix caching) for
   Scenario C's burst-64 throughput workload.

2. **Cross-scenario winner (round 2):** once we have a best-per-scenario
   launcher, compose the best settings into a single config and run A-D
   confirmation sweep.

3. **FP8 weight quantization:** if FP8 KV alone is strong, try full FP8 weight
   quant via `--quantization fp8` (supported on Blackwell). Risk: quality gate
   regression. Requires a clean relaunch and full quality eval.

4. **Speculative decoding for Scenario D:** if r2-tanjiro's bundle works well,
   add n-gram speculation (5 tokens) to the D launcher in round 2 — the
   concurrency=4 burst profile may still benefit.

5. **SGLang as alternative engine:** if vLLM consistently underperforms on
   prefill-heavy scenarios, try SGLang which has different scheduler and prefill
   primitives. The public reference shows SGLang default is competitive
   (3.92× vs vLLM 4.05× aggregate).

6. **Prefix caching (`--enable-prefix-caching`):** the LongBench-v2 requests
   within a scenario share the same system prompt. Enabling prefix caching could
   cut TTFT significantly on repeated requests — investigate if Scenario A or D
   has shared prefix structure.
