# Research Ideas — 2026-05-28 14:45

Generated for branch `ib-20260528-12h-r2`. Active in-flight (do not duplicate): PR #145 (Sc A FP8 KV), PR #146 (Sc B FP8+spec15/lookup8).

---

## Hypothesis 1 — Sc D: Deeper Spec Depth Transfer (spec15/lookup8) + FP8 Composition

**Target:** Scenario D — geomean(1/TTFT.p50, 1/TPOT.p50, req/s), SMAC3 ceiling 5.69x, current best 2.073x

**Mechanism:** PR #141 proved spec15/lookup8 gives 3.550x on Sc B — a +32% jump over spec5/lookup4 — and the gain is superlinear because longer lookup windows find more prefix matches in a 1024-token prompt. PR #139 established that FP8 and n-gram compose near-multiplicatively on Sc D (1.83x TTFT × 2.40x TPOT → 2.073x geomean). Sc D's decode phase is 2048 tokens from a 4096-token prompt, which is longer overlap than Sc B's 8192 from 1024 — the lookup window should find just as many n-gram hits. Upgrading the spec depth from (5, lookup4) to (15, lookup8) in the Sc D composition is the cheapest next lever: it changes two integers and is already validated in a decode-bound scenario.

**Expected magnitude:** ~2.4–2.8x geomean. Rough calculation: TTFT component stays at ~1.83x (spec doesn't help prefill). TPOT component: 2.40x current × ~1.3x from deeper spec (same ratio as Sc B going 2.687→3.550) = ~3.1x. Geomean lifts from 2.073x to approximately 2.4–2.6x.

**Risk:** LOW. Both primitives are confirmed winners. The only new risk is quality — Sc D quality is already tight at 0.953 (τ=0.95). Spec decoding introduces token distribution shift; monitor MMLU-Pro score carefully. If quality fails, fall back to spec12/lookup6 before spec15.

**Launcher flags (vLLM, on top of current Sc D winner):**
```bash
--max-num-seqs 32 \
--max-num-batched-tokens 4096 \
--enable-chunked-prefill \
--no-enable-prefix-caching \
--kv-cache-dtype auto \
--quantization fp8 \
--speculative-config '{"method":"ngram","num_speculative_tokens":15,"prompt_lookup_max":8,"prompt_lookup_min":2}'
```
Run arms: spec15/lookup8 (primary), spec12/lookup6 (quality fallback), spec20/lookup10 (upper bound probe).

---

## Hypothesis 2 — Sc C: SGLang FP8 KV Cache

**Target:** Scenario C — geomean req/s across 3 traffic profiles, SMAC3 ceiling 46.70x, current best 24.305x

**Mechanism:** PR #144 established SGLang+LPM as the Sc C winner at 24.305x using 82.5 GiB VRAM — well below the ~97.9 GiB ceiling. FP8 weight quantization *hurt* Sc C by -3.3% (PR #140) because dequant overhead per forward pass is paid by every batch, not amortized across tokens like in decode. FP8 KV cache is a different intervention: it compresses each KV block from BF16 (2 bytes/element) to FP8 (1 byte/element), allowing ~2x more concurrent KV blocks in the same memory envelope. In a scheduler-bound scenario like Sc C with 256 concurrent requests × 1024 output tokens, KV block capacity is the binding constraint. More KV blocks → lower eviction rate → less recomputation → higher effective throughput. The dequant cost for FP8 KV is lower than for weights because it only touches attention, not the full forward pass.

SGLang FP8 KV is enabled via `--kv-cache-dtype fp8_e5m2` or `--kv-cache-dtype fp8_e4m3`. The Triton attention backend on SGLang already handles FP8 KV on sm89+, and Blackwell (SM120) inherits this path.

**Expected magnitude:** 5–20% throughput gain on Sc C. The 15.2 GiB of freed KV capacity (97.9 − 82.5) could accommodate ~1.5× more concurrent KV state. Not all of that translates to req/s because the scheduler and compute also bound; realistically 8–15% is achievable.

**Risk:** MEDIUM. FP8 KV quality degradation must pass τ=0.95 gate. `e5m2` is more conservative (wider range, less precision) and usually safer for KV cache than `e4m3`. Run `e5m2` first and check MMLU-Pro score before `e4m3`.

**Launcher flags (on top of PR #144 SGLang winner):**
```bash
python launch_sglang.py \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size 8192 \
  --schedule-policy lpm \
  --max-running-requests 256 \
  --attention-backend triton \
  --kv-cache-dtype fp8_e5m2
```
Also push `--mem-fraction-static` from 0.85 to 0.92 to use the freed KV headroom (see Hypothesis 5). Run arms: `fp8_e5m2` (primary) and `fp8_e4m3` (higher compression, more risk).

---

## Hypothesis 3 — Sc D: SGLang + LPM Engine Swap

**Target:** Scenario D — geomean(1/TTFT.p50, 1/TPOT.p50, req/s), SMAC3 ceiling 5.69x, current best 2.073x

**Mechanism:** SGLang beat vLLM by +15.2% on Sc C (PR #144 vs PR #142) primarily via the LPM scheduler's radix-cache prefix reuse. Sc D features 4096-token prompts with conc=4: four concurrent long-context requests create non-trivial prefix overlap potential, especially across repeated benchmark calls. SGLang's radix cache will deduplicate shared prompt prefixes across the 4 concurrent streams, cutting effective prefill FLOPS. Additionally, SGLang's Triton attention backend on Blackwell avoids the FlashAttn `VLLM_ATTENTION_BACKEND=FLASH_ATTN` path that vLLM uses, which may have less-optimized Blackwell kernels.

Compose SGLang + LPM + FP8 weights + spec15/lookup8 on Sc D as a unified hypothesis. This stacks three confirmed (or near-confirmed) winners into a single configuration.

**Expected magnitude:** Uncertain but potentially large — if SGLang provides a 10–15% lift on the req/s component and spec15 provides a 30% lift on TPOT (as on Sc B), the geomean could reach 2.5–3.0x.

**Risk:** MEDIUM-HIGH. This is a larger compositional bet. Each component is validated separately but not together on Sc D. The main risks are: (a) SGLang's spec+FP8 quality on Sc D (already tight at 0.953), (b) CUDA graph conflicts between spec decoding and SGLang's overlap scheduler on conc=4. Staging recommendation: first confirm SGLang BF16 baseline on Sc D, then add FP8, then add spec.

**Launcher flags (SGLang, staged):**
```bash
# Arm 1: SGLang + LPM baseline on Sc D
python launch_sglang.py \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 4096 \
  --schedule-policy lpm \
  --max-running-requests 4 \
  --attention-backend triton

# Arm 2: Add FP8 weights
  --quantization fp8

# Arm 3: Add spec15/lookup8
  --speculative-algorithm NGRAM \
  --num-speculative-tokens 15 \
  --speculative-num-draft-tokens 15
```
Note: SGLang NGRAM spec flag is `--speculative-algorithm NGRAM` not `--speculative-config`. Verify exact flag spelling from `python -m sglang.launch_server --help`.

---

## Hypothesis 4 — Sc C: SGLang NGRAM Speculative Decoding

**Target:** Scenario C — geomean req/s across 3 traffic profiles, SMAC3 ceiling 46.70x, current best 24.305x

**Mechanism:** Sc C has 1024-token decode per request with a 1024-token input. The n-gram lookup window can match against the entire 1024-token input. At conc=256, the overlap scheduler is already on. Adding SGLang NGRAM spec decoding on top of LPM+radix-cache targets the decode throughput component of req/s. The gain mechanism is: each speculated-and-accepted token reduces the number of decode steps per request, so total wall time per request falls, lifting req/s.

The caveat from SpecDecode-Bench is that batch-size matters: at high concurrency (batch=256 effective) the speculative overhead per draft step becomes non-trivial and gains shrink. The constant-load and Poisson profiles in Sc C will naturally have variable batch sizes; the burst profile will hit maximum concurrency most. Net effect is uncertain but non-zero.

SGLang NGRAM disables the overlap scheduler, which may partially offset the spec gain. This must be measured empirically.

**Expected magnitude:** 3–12% req/s gain. Conservative because of the high-concurrency headwind. Most likely beneficial on constant and Poisson profiles, neutral or slightly negative on burst.

**Risk:** MEDIUM. Unknown interaction between NGRAM spec and overlap scheduler in SGLang. Simple to test — one flag addition to the PR #144 base.

**Launcher flags:**
```bash
python launch_sglang.py \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 8192 \
  --schedule-policy lpm \
  --max-running-requests 256 \
  --attention-backend triton \
  --speculative-algorithm NGRAM \
  --num-speculative-tokens 5 \
  --speculative-num-draft-tokens 5
```
Also try `--num-speculative-tokens 10` as a second arm. If overlap scheduler is disabled by NGRAM, also try `--disable-overlap-scheduler False` if the flag is available.

---

## Hypothesis 5 — Sc C: SGLang Memory Fraction Push (0.85 → 0.92)

**Target:** Scenario C — geomean req/s across 3 traffic profiles, SMAC3 ceiling 46.70x, current best 24.305x

**Mechanism:** PR #144 used `--mem-fraction-static 0.85`, consuming 82.5 GiB of 97.9 GiB available VRAM. There is ~15 GiB of headroom that could be allocated to the KV block pool. In Sc C with 256 concurrent requests × 1024 KV positions each, the KV block pool size directly controls how many requests can remain resident without eviction. More KV blocks → fewer evictions → less recomputation → higher effective throughput. This is the simplest possible intervention: one float argument change.

The risk of OOM is managed by staying below 0.93 (keeping ~7 GiB for activations and CUDA context). The practical ceiling for `--mem-fraction-static` on this image is approximately 0.91–0.92 based on typical SGLang activation overhead for 7B models at BF16.

**Expected magnitude:** 3–8% throughput gain by reducing eviction pressure.

**Risk:** LOW. Straightforward flag change. OOM risk at 0.93+, so stay at 0.92. Easily reversible.

**Launcher flags:**
```bash
python launch_sglang.py \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size 8192 \
  --schedule-policy lpm \
  --max-running-requests 256 \
  --attention-backend triton
```
Arms: 0.90, 0.92. Do not exceed 0.92 without confirming no OOM at 256 concurrent requests.

---

## Hypothesis 6 — Sc B: Spec Depth Upper Bound Probe (depth 20/25, lookup_min=1)

**Target:** Scenario B — 1/TPOT.p50, SMAC3 ceiling 15.23x, current best 3.550x (spec15/lookup8)

**Mechanism:** PR #141 found that spec15/lookup8 gives 3.550x and the saturation point was not definitively established. The spec depth sweep went from 5→10→15 and the gains were monotonically increasing. It is not known whether 3.550x is near-peak or whether there is meaningful headroom at depth 20–25. Additionally, `prompt_lookup_min=1` was not tested — allowing matches starting from 1-gram (unigram) is more permissive and may increase accept rate further, especially on Sc B's 1024-token prompt where exact matches may be rare.

At spec depth 20–25, the draft overhead grows (more tokens generated per step), and if the accept rate doesn't scale proportionally, TPOT will degrade. This hypothesis cleanly tests the saturation boundary.

**Expected magnitude:** +5–15% over 3.550x if accept rate scales. Flat or slightly negative if draft overhead dominates. Either result sharpens the model of spec depth trade-offs.

**Risk:** LOW. Simple hyperparameter change, same codebase as PR #141 winner.

**Launcher flags (vLLM, on top of PR #141 winner):**
```bash
# Arm 1: spec20/lookup8/min1
--speculative-config '{"method":"ngram","num_speculative_tokens":20,"prompt_lookup_max":8,"prompt_lookup_min":1}'

# Arm 2: spec25/lookup10/min1
--speculative-config '{"method":"ngram","num_speculative_tokens":25,"prompt_lookup_max":10,"prompt_lookup_min":1}'

# Arm 3: spec15/lookup8/min1 (isolate min effect)
--speculative-config '{"method":"ngram","num_speculative_tokens":15,"prompt_lookup_max":8,"prompt_lookup_min":1}'
```
All arms keep remaining flags identical to PR #141 best arm.

---

## Hypothesis 7 — Sc A: SGLang Prefill Engine Swap

**Target:** Scenario A — 1/TTFT.p50, SMAC3 ceiling 4.48x, current best 1.866x (vLLM FP8)

**Mechanism:** Sc A is single-stream, 8192-token prefill, no meaningful decode. The metric is TTFT. SGLang's Triton backend on Blackwell beat vLLM's FlashAttn backend by a substantial margin on Sc C, likely because the Triton kernels are better tuned for SM120 than the FlashAttn path. A single-stream 8192-token prefill is pure attention compute — the regime where kernel quality matters most. If SGLang's Triton backend executes the attention kernel faster than vLLM's FlashAttn, TTFT will improve.

Additionally, SGLang supports chunked prefill (`--chunked-prefill-size`), which can pipeline prefill chunks with decode and improve throughput — but Sc A is conc=1 so this may not apply. The main bet is Triton kernel quality for long-context prefill.

Compose with FP8 weight quantization (already confirmed on Sc A at 1.866x via vLLM) to preserve that gain.

**Expected magnitude:** 5–20% TTFT reduction if Triton prefill kernels are faster on SM120. Uncertain because Sc C gains were largely from the scheduler/radix cache, not pure kernel speed; kernel gain may be smaller in isolation.

**Risk:** MEDIUM. SGLang + FP8 on Sc A requires verifying: (a) SGLang FP8 quantization flag (`--quantization fp8` or `--quant-type fp8`) loads the same Mistral-7B-FP8 weights correctly, (b) bundled libnuma fix (already documented), (c) realpath PROBLEM_DIR fix (already documented from PR #137).

**Launcher flags:**
```bash
python launch_sglang.py \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 16384 \
  --attention-backend triton \
  --quantization fp8 \
  --max-running-requests 1
```
Compare against vLLM FP8 baseline of 1.866x. Also run SGLang BF16 to isolate engine vs quantization effects.

---

## Hypothesis 8 — Sc D: Larger Max-Num-Seqs Sweep (32 → 64) with FP8+Spec15

**Target:** Scenario D — geomean(1/TTFT.p50, 1/TPOT.p50, req/s), SMAC3 ceiling 5.69x, current best 2.073x

**Mechanism:** PR #139 used `--max-num-seqs 32`. Sc D has conc=4, meaning only 4 requests are active simultaneously. Increasing max-num-seqs beyond 32 may seem unnecessary for conc=4, but it affects how the scheduler batches decode steps: a larger pool allows more speculative token verification in parallel and reduces token-by-token decode serialization. The req/s component of Sc D's geomean is likely bottlenecked by decode throughput, and more batch headroom may help even with low external concurrency.

This is a cheap, minimal-change sweep that could yield a small req/s improvement with zero quality risk.

**Expected magnitude:** 1–5% geomean improvement. Small but compoundable — this is the "free lunch" check before moving to higher-risk ideas.

**Risk:** VERY LOW. Single flag change. No quality impact.

**Launcher flags (vLLM, on top of PR #139 winner):**
```bash
# Arm 1: max-num-seqs=64 (up from 32)
--max-num-seqs 64 \
--max-num-batched-tokens 8192 \
--enable-chunked-prefill \
--no-enable-prefix-caching \
--kv-cache-dtype auto \
--quantization fp8 \
--speculative-config '{"method":"ngram","num_speculative_tokens":15,"prompt_lookup_max":8,"prompt_lookup_min":2}'

# Arm 2: max-num-seqs=128 (aggressive)
--max-num-seqs 128 \
--max-num-batched-tokens 16384
```
Note: `--max-num-batched-tokens` should scale with `--max-num-seqs` to avoid the scheduler filling the batch with tiny decode steps. Scale as 128×max_num_seqs.

---

## Priority Ranking

| Rank | Hypothesis | Scenario | Risk | Expected Gain | Rationale |
|------|-----------|----------|------|--------------|-----------|
| 1 | H1: Sc D spec15/lookup8 depth transfer | D | LOW | +15–30% geomean | Proven mechanism, cheapest next step |
| 2 | H2: Sc C SGLang FP8 KV cache | C | MEDIUM | +8–15% req/s | Large headroom, memory-bound lever |
| 3 | H5: Sc C mem-fraction push 0.92 | C | LOW | +3–8% req/s | One float change, immediate compound |
| 4 | H6: Sc B spec depth upper bound | B | LOW | +5–15% 1/TPOT | Saturation test, diagnostic value |
| 5 | H3: Sc D SGLang+LPM swap | D | MEDIUM-HIGH | +20–40% if it works | High upside, staging reduces risk |
| 6 | H7: Sc A SGLang prefill | A | MEDIUM | +5–20% 1/TTFT | Kernel quality bet, uncertain |
| 7 | H4: Sc C SGLang NGRAM spec | C | MEDIUM | +3–12% req/s | Speculative, batch-size headwind |
| 8 | H8: Sc D max-num-seqs sweep | D | VERY LOW | +1–5% geomean | Cheap free-lunch check |

---

## Notes for Assignment

- H1 (Sc D spec15) should be assigned first — it is the lowest-risk, highest-confidence improvement available.
- H2 (Sc C FP8 KV) and H5 (Sc C mem-fraction) can be combined into one PR since both target Sc C and can be run as arms.
- H3 (Sc D SGLang) requires SGLang boot verification on Sc D first; instruct the student to confirm BF16 baseline before adding FP8+spec.
- H6 (Sc B upper bound) should wait for PR #146 (in-flight) to land before assigning, to avoid duplicate Sc B work.
- All SGLang launchers must include the bundled libnuma fix and the `export PROBLEM_DIR="$(realpath $PROBLEM_DIR)"` workaround from PR #137/#138/#144.
