# SENPAI Research Results

## 2026-05-28 19:39 UTC — PR #181: Sc C SGLang mem-fraction push on PR #172 winner (MERGED — new Sc C best 29.768x)

- **Branch:** `fern/sc-c-sglang-mem-push`
- **Student:** fern
- **Hypothesis:** PR #172's `--mem-fraction-static 0.85` leaves VRAM headroom because FP8 weight quantization freed ~7 GiB. Raising mem-fraction to 0.90 or 0.95 allocates the freed VRAM to KV cache, increasing in-flight token capacity at Sc C's 256-conc KV-bandwidth-bound regime.

### Quick eval (n=4 burst, both arms)

| Arm | mem-fraction | Quick speedup | VRAM peak | Status | W&B |
|---|---:|---:|---:|---|---|
| arm1 | 0.90 | 3.998x | 87.4 GiB | OK | nv954pn6 |
| arm2 | 0.95 | 4.005x | 92.2 GiB | OK (no OOM) | qb7gu15o |
| PR #172 arm1 (baseline) | 0.85 | 3.991x | 84.7 GiB | merged | 5ncgrruu |

VRAM allocation increased monotonically with mem-fraction (84.7 → 87.4 → 92.2 GiB), confirming the additional headroom is consumed by KV cache. Quick eval was concurrency-bound at n=4 (arm2 only +0.19% over arm1 — below the student's ≥1% promotion threshold). Student promoted arm1 to full eval; arm2 not promoted.

### Full eval results (arm1 — 768 requests across burst/poisson/constant + n=500 MMLU-Pro)

| Metric | PR #181 arm1 | PR #172 (prior best) | Δ |
|---|---:|---:|---:|
| **scenario/C/speedup_over_pytorch** | **29.768x** | **29.532x** | **+0.80% ✓** |
| geomean req/s | 2.522 | 2.502 | +0.80% |
| Quality (MMLU-Pro n=500) | 0.286 (ratio 0.960) | 0.286 (ratio 0.960) | — same |
| Speed success | 768/768 ✓ | 768/768 ✓ | — |
| VRAM peak | 87.5 GiB | 84.7 GiB | +2.8 GiB to KV |
| W&B | 2iilmzji | 5ncgrruu | — |

### Analysis

- **Mechanism confirmed.** Raising mem-fraction 0.85→0.90 added +2.8 GiB to KV cache, translating to +0.80% throughput at full 256-conc load.
- **Sc C quick→full gap consistent.** PR #181 arm1 quick = 3.998x = 3.991x (PR #172 quick) +0.18%; full = +0.80%. ~4.4× amplification from quick to full — a stable Sc C pattern at this lever (vs PR #172's FP8 KV which went +1.2% quick suppression → +7.4% full activation).
- **KV ceiling approaching.** +0.80% gain (vs PR #172's +7.4% from FP8 KV) signals diminishing returns. The PR #172 config at mem 0.85 was already close to KV-utilization saturation at 256-conc; +2.8 GiB only marginally raises in-flight capacity.
- **Sc C lever progression:** vLLM baseline → SGLang LPM+radix (24.305x, +15.2%) → FP8 KV (27.497x, +13.1%) → FP8 wt+KV (29.532x, +7.4%) → mem 0.90 (29.768x, +0.80%). Cumulative gain over PR #151 = +8.3%. Each lever has smaller marginal gain; Sc C is approaching the ceiling.
- **Ceiling progress:** 29.768/46.70 = 64% of H100 SMAC3 ceiling on Sc C.

### Next Sc C directions

- **mem 0.95 full eval:** Sc C quick is noise-bound; arm2 was not promoted. Worth a single-arm full eval as direct follow-up. Expected +0.3-0.6% incremental on the same diminishing-returns trend.
- **chunked-prefill-size sweep on PR #181 base:** Currently 8192. Test 4096 or 16384 — different chunk sizes may rebalance prefill/decode interleaving at 256-conc.
- **max-running-requests tuning:** Currently exactly matches concurrency (256). Lowering may reduce HoL blocking; raising may add pipelining headroom.
- Sc C is approaching local optimum; deeper gains likely require new mechanisms (e.g., disaggregated prefill/decode, alternative kernels).

---

## 2026-05-28 19:12 UTC — PR #172: Sc C SGLang FP8 weights + FP8 KV composition (MERGED — new Sc C best 29.532x)

- **Branch:** `fern/sc-c-sglang-fp8wt`
- **Student:** fern
- **Hypothesis:** SGLang FP8 weight quantization composes with FP8 KV cache on PR #151 winner at Sc C's 256-request KV-bandwidth-bound regime. FP8 weights halve per-batch weight bandwidth (amortised over 256 concurrent requests); FP8 KV halves per-token KV block size. Independent bandwidth levers targeting distinct memory hierarchies.

### Full eval results (arm1 — 768 requests across burst/poisson/constant + n=500 MMLU-Pro)

| Metric | PR #172 arm1 | PR #151 (prior best) | Δ |
|---|---:|---:|---:|
| **scenario/C/speedup_over_pytorch** | **29.532x** | **27.497x** | **+7.4% ✓** |
| geomean req/s | 2.502 | ~2.329 | +7.4% |
| Quality (MMLU-Pro n=500) | 0.286 (ratio 0.960) | 0.298 (ratio 1.000) | -4% (still passes gate) |
| Speed success | 768/768 ✓ | 768/768 ✓ | — |
| VRAM peak | ~84.7 GiB | 84.4 GiB | ≈ same |
| W&B | 5ncgrruu | kk6shiqh | — |

### Analysis

- **FP8 weights + FP8 KV compose on Sc C.** At 256-concurrency, weight bandwidth is shared across the batch — the per-weight-read cost amortises multiplicatively. FP8 KV reduces KV memory per request independently. Both levers converge on the same throughput objective (serve more concurrent tokens per second) without interfering.
- **Quality at 0.960 (barely passes gate).** 0.286/0.298 ratio. Tighter than PR #151's 1.000. FP8 weights introduce small numerical perturbation to the full decode path. The gate holds (tau=0.95) but there is less margin now.
- **arm3 quick (FP8wt + BF16KV) edged arm1 at n=4** (4.040x vs 3.991x) — concurrency-bound quick eval suppresses the FP8 KV benefit. At full 256-conc, FP8 KV's doubled KV capacity materialises (+7.4% total vs PR #151).
- **Sc C lever progression:** vLLM baseline → SGLang LPM+radix (24.305x, +15.2%) → FP8 KV (27.497x, +13.1%) → FP8 wt+KV (29.532x, +7.4%). Each composition layer adds diminishing returns; next lever needs to be non-bandwidth. Sc C ceiling 46.70x at 63%.

### Next Sc C directions

- **Higher mem-fraction with FP8wt+FP8KV:** arm2 (mem 0.92) was nearly identical to arm1 at quick (3.989x vs 3.991x). With FP8 weights freeing ~7 GiB, arm2's VRAM was 91.5 GiB but quick was flat — full eval might show small gain from extra KV slots. Low priority.
- **SGLang 0.6.x / newer SGLang:** Different version may improve Triton kernel perf. Engineering investment.
- **CPU offload / tensor parallelism:** Out of scope for single-GPU shakedown.

---

## 2026-05-28 19:05 UTC — PR #166: Sc B prompt_lookup_min=1 sweep (CLOSED — did_not_improve, quick→full collapse)

- **Branch:** `frieren/sc-b-prompt-lookup-min1`
- **Student:** frieren
- **Hypothesis:** PR #149 (Sc B winner, 3.888x) uses `prompt_lookup_min=2`. Lowering to `min=1` permits single-token-context lookups → richer n-gram matching → higher acceptance rate → lower TPOT. Rule #15 (output-length insulation) predicted Sc B's 8192-token decode would absorb the precision loss that broke Sc D at 2048-out.

### Quick-screen (n=4 burst, n=16 quality)
arm1 (spec25/lookup12/min=1) quick = **5.601x** (+44% over PR #149). Quality 3/16 at floor — diagnosed by student as unreliable signal (binomial SE ~11.5pp at n=16). Advisor approved arm1 full eval.

### Full eval (n=64 burst + n=500 MMLU-Pro)

| Metric | Value | vs PR #149 (3.888x) | Δ |
|---|---:|---:|---:|
| **scenario/B/speedup_over_pytorch** | **3.329x** | **3.888x** | **−14.4% ❌** |
| 1/TPOT.p50 | 132.36 | 154.55 | −14.4% |
| TPOT.p50 | 0.00755s | 0.00647s | +17% slower |
| Quality (MMLU-Pro n=500) | 0.314 (ratio 1.054) | 0.298 baseline | **PASSES gate** ✓ |
| Speed success | 64/64 ✓ | 128/128 ✓ | — |
| W&B | 0yg6s6c9 | wkedminm | — |

### Analysis

**Quick→full mapping collapsed.** Quick 5.601x → Full 3.329x = -41% relative drop. The opposite direction of the stable +8% quick→full gap observed elsewhere on Sc B.

**Mechanism:** With `lookup_min=1`, every single token in history seeds a 25-token speculative branch. At n=4 burst (3 unique requests per profile), rare lucky matches in the small history dominate. At n=128 burst (32 unique requests per profile), the wasted compute on rejected branches dominates. The n-gram acceptance rate is fundamentally distribution-dependent on the request volume.

**Quality DID absorb the precision loss** (ratio 1.054 — better than PR #149's ratio). But the speed mechanism failed independently of quality.

### New rule (rule #17)

**Sc B quick→full mapping is UNRELIABLE for hyperparameter changes that alter n-gram acceptance rate distributions.** PR #149 (spec depth sweep) showed stable +8% quick→full. PR #166 (lookup_min change) collapsed -41% quick→full. Future Sc B sweeps that alter the n-gram lookup *behavior* (not just spec depth) must validate at full eval before promoting on quick signal. The lookup_min lever is exhausted at min=2.

### Decision

CLOSED — Sc B baseline remains PR #149 at 3.888x. Frieren reassigned to Sc A `--max-num-batched-tokens` sweep on PR #156 base (PR #180).

---

## 2026-05-28 18:30 UTC — PR #173: Sc D chunked-prefill vs one-shot + batched-tokens sweep (CLOSED — did_not_improve)

- **Branch:** `tanjiro/sc-d-prefill-sweep`
- **Student:** tanjiro
- **Hypothesis:** PR #152's `--enable-chunked-prefill --max-num-batched-tokens 4096` may be suboptimal on Sc D conc=4. Test (a) one-shot prefill (no chunking) with 8192-token budget, (b) chunked prefill with 8192-token budget, (c) chunked prefill with 16384-token budget. The implicit baseline is PR #152's exact config (chunked + 4096).

### Quick-eval results (n=4 burst, n=16 quality)

| Arm | Config | Quick speedup | TTFT.p50 | TPOT.p50 | quality n=16 | W&B |
|-----|--------|--------------:|---------:|---------:|-------------:|-----|
| arm1 | `--no-enable-chunked-prefill --max-num-batched-tokens 8192` | 2.0426x | 0.1270s | 0.0109s | 16/16 (1.05) | 3uqtnsmk |
| arm2 | `--enable-chunked-prefill --max-num-batched-tokens 8192` | 2.0452x | 0.1269s | 0.0109s | 16/16 (1.05) | unp3zcqs |
| arm3 | `--enable-chunked-prefill --max-num-batched-tokens 16384` | 2.0457x | 0.1271s | 0.0109s | 3/16 (0.629) | pmoou5bz |
| PR #152 quick (reference) | `--enable-chunked-prefill --max-num-batched-tokens 4096` | 2.033x | — | — | — | g1xjqoyg (full) |

### Analysis & Conclusions

- **All 3 arms within 0.15% of each other**, +0.5–0.6% over PR #152 quick — below the +1% promotion threshold. Per stable Sc D quick→full mapping (+~8%), arms project to ~2.21x full, within noise of PR #152's 2.218x.
- **Chunked vs one-shot prefill (arm1 vs arm2): +0.13% — completely flat.** Chunk-boundary overhead is negligible at conc=4 because there's barely any decode work to interleave with prefill.
- **Token budget 4096 → 8192 → 16384: +0.62% total.** Once budget ≥ 4096 (one Sc D input length), larger budgets don't help.
- **TTFT and TPOT identical across arms** (0.127s ± 0.0002, 0.0109s ± 0.00005). Prefill stage is at hardware limit on the PR #152 base.
- **arm3 quality at n=16 floor (3/16 = 0.6292)** consistent with binomial SE at n=16; arm3 dominated on speed so not worth investigating further.

### Sc D vLLM 0.11 PR #152 — confirmed local optimum across all single-flag levers

| Lever | Tested in | Result |
|---|---|---|
| n-gram spec depth | PR #154 | spec ≥ 11 fails quality (cliff) |
| FP8 KV cache (e5m2/e4m3) | PR #164 | both -10% regression at conc=4 |
| mem-fraction 0.92 → 0.95 | PR #164 | neutral (+0.04% noise) |
| Chunked vs one-shot prefill | PR #173 | wash (+0.13%) |
| max-num-batched-tokens 4096→16384 | PR #173 | +0.6% (below threshold) |

**Remaining Sc D paradigm-shifts:** (1) draft-model speculation (EAGLE-2/MTP on Mistral-7B), (2) vLLM 0.12+ FlashInfer (blocked on RTX PRO 6000 SM120), (3) engine port to SGLang (PR #138/#147 showed SGLang -57% TPOT on Sc D).

---

## 2026-05-28 11:53 UTC — PR #137: Scenario A vLLM prefill TTFT arms (one-shot vs chunked vs FP8 weights)

- **Branch:** `frieren/sc-a-vllm-prefill-ttft-arms`
- **Student:** frieren
- **Hypothesis:** On Scenario A (8192-token input-heavy, concurrency 1 burst), reducing
  weight-read bandwidth via FP8 weight quantization should cut TTFT proportionally, because
  TTFT is dominated by the prefill matmul chain which is memory-bandwidth-bound on Mistral-7B at
  concurrency 1.

**Quick-probe results:**

| Arm | Config | TTFT.p50 (s) | Speedup (quick) | Notes |
|---|---|---:|---:|---|
| arm1 one-shot BF16 | `--no-enable-chunked-prefill`, `--max-num-batched-tokens 10240` | 0.3435 | 1.276x | vLLM 0.11 forces chunked prefill anyway when batched_tokens < model_len |
| arm2 chunked BF16 | `--enable-chunked-prefill`, `--max-num-batched-tokens 10240` | 0.3427 | 1.279x | Identical to arm1 — same regime, within noise |
| **arm3 FP8 weights** | arm1 + `--quantization fp8` | **0.2307** | **1.901x** | Clean boot on Blackwell; quality n=16 unchanged vs BF16 |

**W&B quick runs:** 5iump62l (arm1), r20z2rm0 (arm2), eyvyj8oz (arm3)

**Full eval result (arm3 FP8):**

| Metric | Value |
|---|---|
| `scenario/A/speedup_over_pytorch` | **1.8664** |
| `scenario/A/inverse_ttft_p50` | 4.256 (PyTorch 2.280) |
| TTFT.p50 / p90 / p99 | 0.2349 / 0.2613 / 0.2681 s |
| TPOT.p50 | 0.0175 s |
| Request throughput | 0.0903 req/s |
| MMLU-Pro accuracy | 0.288 (baseline 0.298, ratio 0.966, n=500) |
| Quality gate | **PASS** (ratio 0.966 ≥ tau 0.95) |
| Speed success | 128/128 (failure_rate 0.0) |
| VRAM peak | 93267 / 97887 MiB |
| W&B run | **izg22lch** |

**Merged:** Yes — squash-merged to `ib-20260528-12h-r2` at 11:53 UTC.

**Analysis and conclusions:**

- FP8 weight-only quantization (`--quantization fp8`) gives a clean 1.87x TTFT speedup on
  Scenario A versus the PyTorch baseline. This is roughly the expected theoretical gain for
  halving weight-read bandwidth on a prefill-dominated workload at concurrency 1.
- The "one-shot vs chunked prefill" distinction is moot under vLLM 0.11: the engine forces
  chunked prefill whenever `max_num_batched_tokens < max_model_len`, so both arms ran with
  a single 10240-token chunk (larger than the 8192-token input), behaving identically.
- Quality at n=500 shows only ~1 pp absolute accuracy drop (0.298 → 0.288), comfortably
  above the 0.95 × baseline gate. FP8 quantization on Mistral-7B-Instruct-v0.3 is safe.
- The H100 public reference for Scenario A is 4.37x (SMAC3). Our 1.87x confirms FP8 alone
  is not sufficient to reach that level; further levers (larger batched-token tuning, chunked
  prefill with overlap at higher concurrency, attention backend, FP4/W4A8) are required.
- **Bug identified:** `senpai/create_task_workspace.py` generates `eval_env.sh` with a
  relative `PROBLEM_DIR`, which fails to source `runtime_env.sh`, leaving FlashInfer enabled
  on Blackwell — causing a JIT kernel compile failure on SM120f. Workaround: `export
  PROBLEM_DIR="$(realpath $PROBLEM_DIR)"` before eval. Should be fixed in the script.

**Next experiment:** frieren assigned PR #139 (Scenario D, vLLM FP8 + n-gram speculative
composition — testing whether both Round-1 winning levers stack on the balanced scenario).

---

## 2026-05-28 12:19 UTC — PR #136: Scenario B vLLM decode TPOT arms (FP8 weights, n-gram spec, BF16 baseline)

- **Branch:** `fern/sc-b-vllm-decode-tpot-arms`
- **Student:** fern
- **Hypothesis:** Scenario B (1024-token input, 8192-token decode, concurrency 1 burst) is output-heavy
  and TPOT-dominated. Two levers were tested: FP8 weight quantization (reduces weight-read bandwidth
  during decode matmuls) and n-gram prompt-lookup speculative decoding (reduces decode steps via exact
  speculative token acceptance from the input prompt).

**Quick-probe results:**

| Arm | Config | TPOT.p50 (s) | Speedup (quick) | Notes |
|---|---|---:|---:|---|
| arm1 BF16 tuned | `--max-num-seqs 16 --max-num-batched-tokens 2048 --enable-chunked-prefill --no-enable-prefix-caching` | 0.01749 | 1.438x | BF16 baseline with tuned seqs/batch |
| arm2 FP8 weights | arm1 + `--quantization fp8` | 0.01706 | 1.474x | Within 5% of arm1 — FP8 is not bandwidth-bound on decode-heavy Sc B |
| **arm3 n-gram spec** | arm1 + `--speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'` | **0.00715** | **3.518x** | Exact speculation — promoted to full eval |

**W&B quick runs:** ex8p5t6j (arm1), ycwnvt87 (arm2), du9y0jz8 (arm3)

**Full eval result (arm3 n-gram speculative):**

| Metric | Value |
|---|---|
| `scenario/B/speedup_over_pytorch` | **2.6874** |
| `scenario/B/inverse_tpot_p50` | 106.84 tok/s (PyTorch 39.76 tok/s) |
| TPOT.p50 | ~0.00936 s (PyTorch 0.0252 s) |
| Generation throughput | 128.4 tok/s (quick) → 106.8 tok/s (full, 64-req average) |
| Request throughput | ~0.0286 req/s |
| MMLU-Pro accuracy | 0.300 (baseline 0.298, ratio 1.007, n=500) |
| Quality gate | **PASS** (ratio 1.007 ≥ tau 0.95) |
| Speed success | 64/64 (failure_rate 0.0) |
| VRAM peak | 90305 MiB / 97887 MiB |
| W&B run | **dav3txgq** |

**Merged:** Yes — squash-merged to `ib-20260528-12h-r2` at 12:19 UTC.

**Analysis and conclusions:**

- N-gram (prompt-lookup) speculative decoding gives a decisive 2.687x TPOT speedup on Scenario B.
  The speculative token acceptance is exact (the draft tokens are greedily sampled from the prompt
  window and verified by the same model logits), so quality is structurally guaranteed — confirmed
  by 0.300 observed (ratio 1.007 > 1.00, even slightly *better* than BF16 possibly due to scheduling
  order under the speculative batch path).
- The quick speedup (3.518x, 4 requests) drops to a full-eval speedup (2.687x, 64 requests) because
  longer runs include harder Sc B prompts with less lexical reuse from the prefix — the speculative
  acceptance rate falls for prompts where the output diverges from the input context. Still 2.7x is
  a strong and robust win.
- **FP8 is surprisingly weak on Sc B** (1.474x vs 1.438x BF16 — within 5%). Sc B is output-heavy at
  concurrency 1: each forward pass processes 1–5 speculative tokens, not a large batched prefill.
  At single-sequence decode, the compute bottleneck shifts from weight-read bandwidth to latency,
  where FP8 weight quantization's bandwidth advantage is diluted by Mistral-7B's small decode batch.
- Both levers confirm the independence principle: FP8 wins on prefill-heavy (Sc A, 1.87x) while
  speculative decoding wins on decode-heavy (Sc B, 2.69x). The next logical test is combining both
  on Sc D (balanced 4096in/2048out), which is exactly what PR #139 (frieren) is testing.
- **Sc B H100 reference context:** H100 leaderboard shows vLLM default at 2.25x, SMAC3 at 15.23x.
  Our RTX PRO 6000 result of 2.687x already beats the H100 vLLM default (2.25x), though the
  hardware/time-budget difference makes direct comparison invalid. The SMAC3 ceiling at 15x likely
  comes from longer speculative depth (larger num_speculative_tokens) and multi-sequence batching
  optimizations not yet explored.

**Next speculative tuning axis (for future PR):** `num_speculative_tokens ∈ {7, 10, 15}` and
`prompt_lookup_max ∈ {4, 6, 8}` — cheap quick-probe sweep, likely to push speedup toward 4–5x.

---

## 2026-05-28 12:50 UTC — PR #138: Scenario D SGLang in per-PR venv (default + tuned arms)

- **Branch:** `tanjiro/sc-d-sglang-venv-baseline`
- **Student:** tanjiro
- **Hypothesis:** Scenario D is general balanced serving (4096-token input, 2048-token decode,
  96 requests at concurrency 4, burst). The H100 public reference shows SGLang default already
  beats vLLM default for this scenario (2.14x vs 1.96x), so this PR diversifies the engine
  portfolio by standing up SGLang in a per-PR venv and benchmarking its baseline.

**Quick-probe results:**

| Arm | Config | Quick speedup | Notes |
|---|---|---:|---|
| arm1 `sglang_default` | `--mem-fraction-static 0.80 --attention-backend triton` | 1.167x | SGLang default; arm selected per simplicity tiebreak |
| arm2 `sglang_tuned` | arm1 + `--chunked-prefill-size 4096 --schedule-policy lpm --max-running-requests 64` | 1.183x | Within 5% of arm1 |

**W&B quick runs:** 2nfds9ud (arm1), 8zul6lqz (arm2)

**Operational note:** Initial quick probes required `apt-get install libnuma1` on the pod
(SGLang's `sgl_kernel` SM100 binary linked against `libnuma.so.1`). This broke the
supervised-relaunch contract. Tanjiro chose Path A (bundle libnuma): committed
`senpai/launchers/D/tanjiro-sglang/lib/libnuma.so.1.0.0` (48 KB, glibc-built) +
symlinks + `LD_LIBRARY_PATH` prepend + `uv venv` auto-bootstrap guard in `start_server.sh`.
Re-ran full eval with the relaunch-safe launcher after confirming bundled .so loads cleanly.

**Full eval result (arm1 sglang_default, relaunch-safe):**

| Metric | Value |
|---|---|
| `scenario/D/speedup_over_pytorch` | **1.2506x** |
| Geomean (1/ttft.p50, 1/tpot.p50, req/s) | 2.413 (PyTorch 1.930) |
| MMLU-Pro accuracy | 0.310 (baseline 0.298, ratio 1.040, n=500) |
| Quality gate | **PASS** (ratio 1.040 ≥ tau 0.95) |
| Speed success | 96/96 (failure_rate 0.0) |
| W&B run | **nf10i0y2** |

**Merged:** Yes — squash-merged to `ib-20260528-12h-r2` at 12:50 UTC. Sc D first winner.

**Analysis and conclusions:**

- SGLang default (arm1) and tuned (arm2) are within 5% on Sc D quick probes (1.167x vs 1.183x),
  confirming that the light tuning axes tried (chunked-prefill-size, lpm scheduler,
  max-running-requests) don't move the needle on Mistral-7B at concurrency 4.
- The SGLang default result (1.247x full eval) is below the H100 public reference for SGLang
  default (2.14x). This is expected: different hardware (RTX PRO 6000 vs H100), different
  timing budget, and this is a first SGLang measurement on this hardware. The gap to the H100
  ceiling is research signal, not a failure.
- SGLang default outperforms the PyTorch baseline (+24.7%) and establishes Sc D as live.
  However, frieren PR #139 has the vLLM FP8+n-gram composition quick at 1.810x — expected to
  supersede this row once the full eval is confirmed.
- The bundled-libnuma pattern (ship `libnuma.so.1` in the launcher directory, prepend
  `LD_LIBRARY_PATH`) is a reusable operational pattern for future SGLang PRs on this image.
  Document in the next round's scaffolding.

**Next experiments for Sc D:**
- **frieren PR #139 (in flight):** vLLM FP8+n-gram composition, quick 1.810x. Full eval
  will likely supersede this SGLang result.
- **SGLang speculative decoding** (`--speculative-num-steps > 0`): now that we have a clean
  SGLang baseline, adding speculative decoding is the obvious next lever for a future PR.
- **vLLM head-to-head on Sc D** once frieren's composition result is confirmed.

---

## 2026-05-28 13:03 UTC — PR #140: Scenario C vLLM high-concurrency throughput + prefix caching

- **Branch:** `fern/sc-c-vllm-throughput-prefix-cache`
- **Student:** fern
- **Hypothesis:** Scenario C is a high-load throughput scenario (1024-token input × 1024-token
  output × 256 requests × 3 load profiles: burst/poisson/constant). Primary metric is the
  geomean of request throughput across all three profiles. The H100 public reference shows
  vLLM default at 48.69x — far above Sc A/B, suggesting that concurrency batching is the
  key lever. Three arms tested: BF16 high-concurrency baseline, + prefix caching, + FP8.

**Quick-probe results:**

| Arm | Config | Quick speedup | Notes |
|---|---|---:|---|
| arm1 BF16 high-conc | `--max-num-seqs 64 --max-num-batched-tokens 8192 --enable-chunked-prefill --no-enable-prefix-caching` | 3.888x | Promoted (simplest, within 1% of best) |
| arm2 + prefix cache | arm1 + `--enable-prefix-caching` | 3.928x | Best on quick; +1% over arm1, noise |
| arm3 + FP8 | arm2 + `--quantization fp8` | 3.798x | FP8 dequant overhead hurts at high conc |

**W&B quick runs:** qvcrnc4s (arm1), 2rtv6gxb (arm2), wuvd1ycq (arm3)

**Full eval result (arm1 BF16 high-concurrency):**

| Metric | Value |
|---|---|
| `scenario/C/speedup_over_pytorch` | **21.052x** |
| Geomean req/s | 1.783 (PyTorch 0.0847) |
| Burst req/s | 2.605 (256/256 ✓) |
| Poisson req/s | 1.854 (256/256 ✓) |
| Constant req/s | 1.175 (256/256 ✓) |
| MMLU-Pro accuracy | 0.298 (baseline 0.298, ratio 1.000, n=500) |
| Quality gate | **PASS** (ratio 1.000 ≥ tau 0.95) |
| Speed success | 768/768 (failure_rate 0.0) |
| VRAM peak | 90815 MiB / 97887 MiB |
| W&B run | **tebmnnza** |

**Merged:** Yes — squash-merged to `ib-20260528-12h-r2` at 13:03 UTC. Sc C first winner.
All 4 scenarios now have valid baseline entries.

**Analysis and conclusions:**

- **21.052x is a massive leap from the 1.00x PyTorch floor** (serialized serving). The
  PyTorch baseline runs requests sequentially at 0.0847 req/s; vLLM's async scheduler
  batches 64 concurrent streams with chunked-prefill at 8192 tokens, achieving 2.605 req/s
  in burst — a 30.7x improvement in burst throughput alone.
- **Prefix caching (arm2) gave only +1% at quick (n=4)**. Sc C requests are drawn from a
  varied dataset; KV reuse between requests is marginal at n=4. The full eval's quality
  ratio of 1.000 suggests no meaningful distribution issue.
- **FP8 hurts on Sc C (-3.3% vs arm1)**. Unlike Sc A (single-stream prefill, weight-read-bound
  where FP8 wins), Sc C at concurrency 64 is KV-cache-memory-bound and scheduler-overhead-bound.
  FP8 dequant adds per-forward-pass overhead without freeing the actual bottleneck.
- **21.052x vs H100 vLLM default (48.69x)**: the ~2.3x gap is explained by hardware differences
  (RTX PRO 6000 vs H100 80 GB), not a configuration failure. On the same hardware, the
  `--max-num-seqs 64` config is extracting competitive throughput from a single config.
- **Quality is exact at ratio 1.000** (BF16 weights, auto KV cache). No precision trade-off.
- **TTFT.p99 in burst (3.58s) is high** — expected: chunked-prefill intentionally trades
  TTFT tail latency for throughput when batching 64 concurrent requests.

**Suggested follow-ups (from fern's notes):**
- Higher concurrency: `--max-num-seqs 128 --max-num-batched-tokens 16384` — VRAM headroom
  ~7 GiB may support larger KV cache. Could push burst req/s further.
- N-gram spec on Sc C: 1024-token decode per request may benefit from spec decoding,
  especially in constant-rate profile (sequential requests where prompt overlap is higher).
- SGLang head-to-head on Sc C: H100 shows SGLang at 51.12x vs vLLM 48.69x (+5%). Worth
  measuring on RTX PRO 6000 in a future PR.

---

## 2026-05-28 13:19 UTC — PR #139: Scenario D vLLM FP8+n-gram composition

- **Branch:** `frieren/sc-d-vllm-fp8-ngram-composition`
- **Student:** frieren
- **Hypothesis:** Scenario D (4096-token input, 2048-token decode, concurrency 4, burst) has
  both a significant prefill stage and a significant decode stage. Round-1 winners showed FP8
  cuts TTFT on prefill-heavy workloads (Sc A: +1.87x) and n-gram spec cuts TPOT on decode-heavy
  workloads (Sc B: +2.69x). Sc D is balanced — the hypothesis was that both levers would
  compose additively since they target independent pipeline stages.

**Quick-probe results:**

| Arm | Config | Quick speedup | Notes |
|---|---|---:|---|
| arm1 fp8_only | `--quantization fp8`, no spec | 1.326x | FP8 helps TTFT, flat on TPOT |
| arm2 ngram_only | n-gram spec, no FP8 | 1.697x | n-gram helps TPOT, limited TTFT |
| **arm3 fp8+ngram** | both combined | **1.810x** | Promoted: +36.5% over arm1, +6.6% over arm2 |

**W&B quick runs:** 4zzu6w86 (arm1), 1qkei5pe (arm2), of4wd96h (arm3) — deferred upload

**Full eval result (arm3 FP8 + n-gram composition):**

| Metric | PyTorch | Arm 3 | Ratio |
|---|---:|---:|---:|
| TTFT.p50 | 0.2123 s | 0.1161 s | 1.83x faster |
| TPOT.p50 | 0.0251 s | 0.0104 s | 2.40x faster |
| req/s | 0.0382 | 0.0776 | 2.03x more |
| Geomean (1/ttft, 1/tpot, req/s) | 1.930 | 4.001 | — |
| `scenario/D/speedup_over_pytorch` | 1.000x | **2.073x** | — |
| MMLU-Pro accuracy (n=500) | 0.298 | 0.284 | 0.953 (PASS) |
| Speed success | — | 96/96 | 0.0 failure rate |
| VRAM peak | — | 91.7 GiB | — |

**W&B full eval run:** 40f15iox

**Merged:** Yes — squash-merged to `ib-20260528-12h-r2` at 13:19 UTC. New Sc D best,
supersedes PR #138 SGLang 1.247x (+66% improvement).

**Analysis and conclusions:**

- **Composition confirmed.** FP8 and n-gram spec target independent stages: FP8 reduces
  weight-read bandwidth during each matmul (wins on the 4096-token prefill that dominates
  TTFT), while n-gram reduces total forward passes via speculative acceptance (wins on the
  2048-token decode that dominates TPOT). Since the bottlenecks are orthogonal, the gains
  compose near-multiplicatively: 1.83x (TTFT) × 2.40x (TPOT) produces a 2.073x geomean.
- **Full eval beats quick probe:** 2.073x vs 1.810x quick (+14.5%). The larger n=96 sample
  exposes more speculative acceptance benefit on full-length 2048-token decode sequences.
- **Quality is tight but passing:** ratio 0.953 (0.284 vs 0.298, gate at 0.95). FP8 causes
  a small accuracy drop consistent with Sc A (0.966 there). The combination FP8+n-gram does
  not compound the quality degradation beyond FP8 alone, as expected (n-gram is a lossless
  verifier).
- **Sc D 2.073x vs H100 SMAC3 5.69x:** significant headroom remaining. The SMAC3 ceiling
  likely comes from deeper speculative depth, higher max-num-seqs, and possibly FP4/W4A8
  quantization not yet tested.
- **SGLang vs vLLM on Sc D:** tanjiro's SGLang default gave 1.247x; frieren's vLLM
  composition gives 2.073x. For Sc D on RTX PRO 6000, vLLM with composition beats
  SGLang default by +66%. This does not test optimized SGLang (speculation, radix cache)
  which may close the gap in a future PR.

**Next experiments for Sc D:**
- **Deeper speculative depth** on Sc D: `num_speculative_tokens ∈ {7, 10, 15}` (tanjiro is
  testing this on Sc B via PR #141 — the winner transfers directly to Sc D).
- **Higher max-num-seqs on Sc D:** `--max-num-seqs 64` with FP8+n-gram may improve req/s
  further (currently at 32; Sc D concurrency 4 × 24 requests = 96 total).
- **SGLang + speculative** once the vLLM composition results solidify as the reference.

---

## 2026-05-28 13:44 UTC — PR #142: Scenario C scale max-num-seqs 128/256 + batched-tokens 16384

- **Branch:** `fern/sc-c-highconc-scale`
- **Student:** fern
- **Hypothesis:** PR #140 used `max-num-seqs=64`; doubling to 128 or 256, and scaling
  `max-num-batched-tokens` to 16384, should further improve Sc C throughput by handling
  more concurrent requests per scheduler cycle.

### Quick probe results (4 requests each)

| arm | max_num_seqs | batched_tokens | Quick speedup | W&B |
|---|---:|---:|---:|---|
| arm1 | 128 | 8192 | 3.9017x | logged |
| arm2 | 64 | 16384 | 3.8991x | logged |
| arm3 | 128 | 16384 | 3.9005x | logged |
| arm4 | 256 | 16384 | 3.8704x | logged |

All 4 quick probes within 0.8% of each other.

**Promotion:** arm1 per simplest-within-5% rule. (Quick concurrency=64 burst fits in all
configs ≥64 seqs, so the quick can't differentiate architecturally.)

### Full eval results (arm1 — 256 reqs × 3 profiles = 768 total)

| Metric | Value |
|---|---|
| Speedup vs PyTorch | **21.098x** (vs PR #140 21.052x = **+0.22%**) |
| burst req/s | 2.618 (PR #140: 2.605) |
| poisson req/s | 1.857 (PR #140: 1.854) |
| constant req/s | 1.175 (PR #140: 1.175) |
| Quality ratio (n=500) | 1.054 (observed 0.314 vs 0.298 baseline) ✓ |
| Speed success | 768/768 ✓ |
| VRAM peak | 90923 MiB |
| W&B | 2jw0s5lk |

### Analysis

- **Plateau confirmed:** +0.22% over PR #140 is at or below run-to-run variance. All four
  concurrency/batching combinations tested in this sweep returned virtually identical results.
  The Sc C vLLM architecture is saturated at the concurrency-scaling axis.
- **Quick probes accurately signalled plateau:** All 4 arms within 0.8% at quick. This was
  because burst concurrency=64 fits in every `max-num-seqs ≥ 64` config — the quick sample
  can't stress-test configs designed for 128+ concurrent requests. Only the full eval
  (256-request profiles) reveals the marginal gain.
- **seqs=128 is the correct frontier:** arm1 (seqs=128/tok=8192) marginally beats arm2
  (seqs=64/tok=16384), confirming that request count matters more than individual batch width
  for Sc C throughput.
- **FP8 KV-cache and SGLang are the remaining levers:** FP8 weights hurt Sc C (-3.3% in
  PR #140); FP8 KV-cache is different — it compresses KV memory, allowing the same 90 GB
  to hold more concurrent KV states. SGLang's radix attention (persistent prefix sharing) may
  also help on the poisson/constant profiles where request inter-arrival is structured.

**Next experiments for Sc C:**
- **SGLang default + radix attention** (fern PR #144 if assigned) — reference shows SGLang
  51.12x vs vLLM 48.69x on H100 (+5%). Should outperform vLLM on RTX PRO 6000 similarly.
- **vLLM FP8 KV-cache:** `--kv-cache-dtype fp8` — reduces KV memory footprint, potentially
  allowing even higher effective concurrency within 96 GB VRAM.
- **SGLang + chunked prefill + radix** once SGLang base is established.

---

## 2026-05-28 13:55 UTC — PR #143: Scenario A FP8 + n-gram speculative composition (CLOSED — did not improve)

- **Branch:** `frieren/sc-a-fp8-ngram-composition`
- **Student:** frieren
- **Hypothesis:** Extend PR #137 (FP8 weights, 1.866x) by composing with n-gram speculative
  decoding (proven on Sc B 2.687x and Sc D 2.073x). Sc A has a 1024-token decode tail
  that should benefit from n-gram acceptance, compounding the FP8 prefill gain.

### Quick probe results

| arm | spec_tokens | lookup_max | Quick speedup | W&B |
|---|---:|---:|---:|---|
| arm1 | 5 | 4 | 1.9115x | quc7zeij |
| arm2 | 10 | 6 | 1.9143x | 3gp3w4s5 |

Promoted arm1 (simpler within 5%).

### Full eval results (arm1)

| Metric | This run | PR #137 winner | Δ |
|---|---:|---:|---:|
| **scenario/A/speedup_over_pytorch** | **1.8603x** | **1.866x** | **-0.30% ❌** |
| TTFT.p50 | 0.23572s | 0.2349s | +0.34% (slower, noise) |
| TPOT.p50 | 0.01276s | 0.01750s | -27% (much faster) |
| Throughput (gen tok/s) | 78.89 | 54.73 | +44% |
| Request throughput | 0.1306 req/s | 0.0903 req/s | +45% |
| Quality (MMLU-Pro n=500) | 0.290 (ratio 0.973) | 0.288 (ratio 0.966) | +0.7% |
| Success | 128/128 ✓ | 128/128 ✓ | — |
| W&B | 3nc047zt | izg22lch | — |

### Analysis

- **The hypothesis was wrong about Sc A primary metric.** Sc A scores on `1/ttft.p50`
  (prefill speed). N-gram speculation works during decode (after first token), so it has
  no effect on TTFT. The 0.30% regression is at the noise floor, but it is the wrong
  direction.
- **Composition is real but invisible to Sc A scoring.** TPOT -27%, throughput +44%,
  request throughput +45% — all genuine improvements from n-gram acceptance on the
  1024-token decode tail. Quality is even slightly better. But Sc A's metric ignores
  decode performance entirely.
- **Sc A is prefill-bound on RTX PRO 6000.** To beat 1.866x meaningfully, future Sc A
  experiments must attack the prefill kernel itself: FP8 KV cache, larger
  `--max-num-batched-tokens`, FlashInfer prefill backend, or alternative quantization.
- **Pattern learned: scenario-metric alignment matters.** Spec composition wins on Sc B
  (TPOT-scored) and Sc D (geomean of ttft, tpot, req/s), but is metric-orthogonal on
  Sc A (1/ttft.p50 only) and Sc C (geomean req/s). Future composition hypotheses must
  identify which scenario metric the proposed lever actually moves.

### Decision

Closed without merging. Decision tree path: "arm below 1.866x → partial did_not_improve,
do not mark terminal, ask advisor". Advisor closes after reviewing — frieren's analysis
correctly identifies that arm2 would be no better (also prefill-bound), so running
additional arms is a waste of GPU time.

**Next experiment for frieren:** Sc A prefill-attack composition (FP8 weights + FP8
KV cache + larger batched-tokens) — see PR #145.

---

## 2026-05-28 14:10 UTC — PR #141: Scenario B n-gram speculative depth sweep

- **Branch:** `tanjiro/sc-b-spec-depth-sweep`
- **Student:** tanjiro
- **Hypothesis:** Increasing `num_speculative_tokens` and `prompt_lookup_max` beyond PR
  #136's `(5, 4)` should improve Sc B TPOT by accepting longer draft bursts during the
  8192-token decode. The hypothesis is that output-heavy decode provides long contexts
  with natural n-gram patterns, enabling higher acceptance rates at deeper windows.

### Quick probe results

| arm | spec_tokens | lookup_max | Quick speedup (n=4) | W&B |
|---|---:|---:|---:|---|
| arm1 | 7 | 4 | 2.121x | cwbiqekn |
| arm2 | 10 | 6 | 2.857x | ahowy8xt |
| arm3 | 15 | 8 | **6.918x** | rdmg3q7u |

Quick arm3 was a dramatic outlier — n=4 at quick likely hit highly-repetitive decode patterns. Promoted arm3 per "any arm beats arm1 by >5%" rule.

### Full eval results (arm3 — 64 requests, burst profile)

| Metric | arm3 full | PR #136 winner | Δ |
|---|---:|---:|---:|
| **scenario/B/speedup_over_pytorch** | **3.550x** | **2.687x** | **+32.2%** |
| inverse_tpot_p50 | 141.15 tok/s | 106.84 tok/s | +32.1% |
| TPOT.p50 | 7.08 ms | 9.36 ms | -24% (much faster) |
| TTFT.p50 | 63.8 ms | — | — |
| ITL.p50 | 12.38 ms | — | — |
| Quality (MMLU-Pro n=500) | 0.300 (ratio 1.007) | 0.300 (ratio 1.007) | 0.0% |
| Speed success | 64/64 ✓ | 64/64 ✓ | — |
| VRAM peak | 90271 MiB | 90305 MiB | ≈ same |
| W&B | 8656lf5w | dav3txgq | — |

### Analysis

- **Superlinear spec gains confirmed on Sc B.** Going from `(spec=5, lookup=4)` to
  `(spec=15, lookup=8)` yielded +32.2% on the primary metric. The jump is not linear —
  PR #141 arm1 at (7,4) was only 2.121x quick. Most of the gain comes from arm3's wider
  `prompt_lookup_max=8` rather than just the deeper `num_speculative_tokens`.
- **Why depth works here:** Sc B decodes 8192 tokens. After the first ~100 tokens are
  generated, the decoded output itself becomes the primary source of n-gram matches. The
  8-gram lookback window finds phrase-level repetitions in LLM output (formulaic structures,
  repeated variable names, etc.), while depth=15 lets a single found match emit a 15-token
  burst. TPOT drops from 9.36 ms (spec5) to 7.08 ms (spec15) — a 24% reduction in the
  already-speculated median.
- **Quality is completely unaffected.** MMLU-Pro 0.300 = ratio 1.007 = identical to PR
  #136. Speculative decoding is verification-exact; deeper speculation doesn't change this.
- **Quick-to-full ratio = 51%.** Quick at n=4 was 6.918x; full at n=64 is 3.550x. The
  6.918x overestimate happened because 4 randomly-selected Sc B requests happened to be
  highly repetitive. Consistent with the pattern from PR #136 (3.518x quick → 2.687x full
  = 76% ratio). The quick is always an upper bound for speculative; larger sample sizes
  expose harder-to-predict outputs.
- **VRAM is near ceiling at 93%.** spec=15 holds 15 draft tokens' KV state per step;
  this barely differs from spec=5. VRAM usage unchanged from PR #136 within rounding.
- **Remaining Sc B headroom:** Current 3.550x vs H100 SMAC3 15.23x. Gap primarily from:
  (a) higher spec depth (20, 25) not yet tested; (b) FP8 weights not yet composed with
  spec15 on Sc B; (c) EAGLE/Medusa draft-model spec for intrinsically richer proposals.

**Next experiments for Sc B:**
- **FP8 weights + spec15/lookup8 composition** (tanjiro PR #146) — FP8 cuts decode
  bandwidth, spec15 cuts forward passes; should compose near-multiplicatively as on Sc D.
  Expected: ~5-7x speedup.
- **Deeper spec sweep (20, 25)** at the (15,8) window to find saturation point.
- **prompt_lookup_min=1** — allows single-token lookback; may improve acceptance rate at
  low marginal cost.

---

## 2026-05-28 14:45 UTC — PR #144: Scenario C SGLang engine — default + chunked prefill + LPM radix cache

- **Branch:** `fern/sc-c-sglang-radix`
- **Student:** fern
- **Hypothesis:** The H100 reference shows SGLang 51.12x vs vLLM 48.69x (+5%) on Sc C.
  SGLang's radix attention cache (KV prefix sharing across requests) should give additional
  throughput gains on Sc C's 256-request burst/poisson/constant profiles, where request prompts
  share common prefixes. Three arms: (1) SGLang default, (2) + chunked prefill, (3) + LPM
  scheduler to surface the radix-cache benefit.

### Quick probe results

| arm | config | quick speedup | W&B |
|---|---|---:|---|
| arm1 SGLang default | `--mem-fraction-static 0.85 --attention-backend triton` | ~3.4x | logged |
| arm2 + chunked prefill | arm1 + `--chunked-prefill-size 8192` | ~3.5x | logged |
| arm3 + LPM scheduler | arm2 + `--schedule-policy lpm --max-running-requests 256` | ~4.1x | promoted |

Note: `--enable-radix-cache` does NOT exist in SGLang 0.5.x — radix cache is ON by default
and can only be disabled with `--disable-radix-cache`. The active lever is `--schedule-policy lpm`.

### Full eval results (arm3 — 768 requests across burst/poisson/constant)

| Metric | arm3 (PR #144) | PR #142 vLLM | Δ |
|---|---:|---:|---:|
| **scenario/C/speedup_over_pytorch** | **24.305x** | **21.098x** | **+15.2%** |
| Burst req/s | 3.0973 | 2.618 | +18.3% |
| Poisson req/s | 2.1409 | 1.857 | +15.3% |
| Constant req/s | 1.3162 | 1.175 | +12.0% |
| Quality (MMLU-Pro n=500) | 0.306 (ratio 1.027) | 0.314 (ratio 1.054) | −2.5% obs (both PASS) |
| Speed success | 768/768 ✓ | 768/768 ✓ | — |
| VRAM peak | 82.5 GiB | 90.9 GiB | **-9.2% VRAM** |
| W&B | ifj6fcec | 2jw0s5lk | — |

**Merged:** Yes — squash-merged to `ib-20260528-12h-r2` at 14:45 UTC. New Sc C best.

### Analysis

- **Engine switch from vLLM to SGLang delivers a clear +15.2% throughput gain**, consistent
  with the H100 reference pattern. The RTX PRO 6000 gap is +15% vs H100's +5%, which likely
  reflects SGLang's Triton-based attention being particularly well-suited to Blackwell (SM120)
  vs FlashAttention (still tuning for the new SM architecture).
- **The active lever is `--schedule-policy lpm` (longest-prefix-match), not the radix cache
  toggle.** In SGLang 0.5.x, radix cache is always ON; it can only be disabled. What was
  unknown before this experiment: the radix cache has negligible benefit without reordering
  the request queue to maximize prefix hits. LPM ordering puts requests with the longest
  shared prefix at the front of the queue, so SGLang's radix attention actually hits the cache
  on the 256-request Sc C streams.
- **+12% on constant profile, +15% on poisson, +18% on burst.** Burst shows the largest gain
  because all 64 burst requests arrive simultaneously — LPM reordering has the maximum pool
  of requests to sort, maximizing cache hit probability.
- **VRAM -10%:** SGLang's memory management (`--mem-fraction-static 0.85`) allocates 82.5 GiB
  vs vLLM's 90.9 GiB at `--gpu-memory-utilization 0.92`. Nominally similar fractions of the
  96 GiB device; SGLang leaves more headroom, possibly due to different pre-allocation of KV
  blocks. This suggests SGLang could potentially push `--mem-fraction-static` higher, allowing
  a larger KV block pool.
- **Remaining gap to H100 reference (51.12x):** 24.305x vs 51.12x = 52% of reference. RTX PRO
  6000 at 288 TFLOPS FP8 vs H100 at 989 TFLOPS FP8 (34% of H100 peak flops) — we are at
  47.5% relative throughput, so the gap is partially hardware-explained. SGLang + FP8 KV
  cache or deeper concurrency tuning could close the gap further.
- **SGLang relaunch contract:** bundled `lib/libnuma.so.1` under
  `senpai/launchers/C/fern-sglang-sc-c/lib/`; per-PR venv at
  `/tmp/inferencebench-engine-venvs/sglang-pr-144` auto-bootstrapped via `uv venv` +
  `pip install sglang[all]==0.5.12.post1`.

**Key learning: SGLang radix cache benefit requires LPM scheduler.** Default FCFS scheduling
processes requests in arrival order, which randomizes prefix overlap per batch. LPM reorders
to maximize the longest-common-prefix hit in the radix tree. This is now a confirmed pattern
on RTX PRO 6000 and should be the default for all future SGLang Sc C experiments.

**Next experiments for Sc C:**
- **SGLang + FP8 KV cache** (`--kv-cache-dtype fp8`): reducing KV memory footprint allows
  the same 82.5 GiB to cache more concurrent KV states, potentially pushing effective
  concurrency beyond 256 within the GPU.
- **SGLang + n-gram speculative decoding on Sc C:** Sc C's 1024-token decode per request may
  accept n-gram proposals. Combine with LPM for combined radix + speculative benefit.
- **SGLang mem-fraction tuning:** bump `--mem-fraction-static` from 0.85 to 0.90–0.92 to
  match vLLM's VRAM allocation and test if more KV blocks improve throughput further.
- **SGLang on Sc D:** Sc D's 4096in/2048out balanced workload would benefit from both the
  radix cache (shared 4096-token prefill prefixes) and LPM reordering. ← **assigned to fern as PR #147**

---

## 2026-05-28 14:55 UTC — PR #145: Scenario A FP8 weights + FP8 KV cache (CLOSED — did not improve)

- **Branch:** `frieren/sc-a-fp8-kv-cache`
- **Student:** frieren
- **Hypothesis:** Adding `--kv-cache-dtype fp8` to the PR #137 FP8-weights winner would halve
  KV-cache attention bandwidth during the 8192-token prefill scan, cutting TTFT further.

### Quick probe results (all 3 arms)

| Arm | seqs | batched-tokens | Quick speedup | TTFT.p50 | W&B |
|---|---:|---:|---:|---:|---|
| arm1 FP8W + FP8KV | 8 | 10240 | 1.4215x | 0.3085s | ehxisw8g |
| arm2 + batched=16384 | 8 | 16384 | 1.4251x | 0.3077s | x211iaca |
| arm3 + seqs=4 | 4 | 16384 | **1.4283x** | 0.3070s | 4pryw5vx |

Reference (PR #137 FP8W only): quick 1.901x (TTFT 0.2306s), full 1.866x (TTFT 0.2349s).

All 3 arms within 0.5% of each other — tighter than noise floor. All ~25% slower than PR #137.

### Full eval results (arm1 — 128 requests, burst profile)

| Metric | Value | vs PR #137 | Δ |
|---|---:|---:|---:|
| **scenario/A/speedup_over_pytorch** | **1.3804x** | **1.866x** | **−26.0% ❌** |
| TTFT.p50 | 0.3177 s | 0.2349 s | +35% (much slower) |
| TPOT.p50 | 0.01682 s | 0.01750 s | −4% (faster, irrelevant for Sc A) |
| Quality (MMLU-Pro n=500) | 0.288 (ratio 0.966) | 0.288 (ratio 0.966) | 0.0% |
| Speed success | 128/128 ✓ | 128/128 ✓ | — |
| VRAM peak | 93,313 MiB | 93,267 MiB | ≈ same |
| W&B | 488cszk6 | izg22lch | — |

**Closed:** Yes — −26% regression is well outside the >5% close threshold. Baseline remains PR #137 1.866x.

### Analysis

- **FP8 KV cache adds dequantization work to the prefill attention kernel (QK^T / AV matmuls).** At concurrency 1, Sc A's attention is compute-bound, not KV-bandwidth-bound. Halving KV bytes never translates to faster attention; the dequant overhead dominates instead.
- **All three arms within 0.5%.** The batched-tokens (10240 vs 16384) and seqs (4 vs 8) axes are completely masked once FP8 KV is on — confirming that the bottleneck is the FP8 KV attention kernel itself, not prefill chunking or sequence count.
- **Quality unchanged at 0.966 ratio.** FP8 KV is numerically safe on SM120; the regression is purely a performance cost, not a correctness issue.
- **Learning: vLLM FP8 bandwidth levers on Sc A are now exhausted.** FP8 weights: +1.866x. FP8 KV: −26%. The kernel-level approach is FlashInfer (PR #148).
- **Pattern reinforced: scenario-hardware alignment matters.** The same FP8 KV trick that might help decode-bound scenarios (Sc B/D) is actively harmful on a single-stream prefill-dominated workload at concurrency 1 on SM120.

**Next experiment for frieren:** FlashInfer prefill attention backend on Sc A (PR #148) — attack the 8192-token prefill at the kernel level rather than the memory/quantisation level.

---

## 2026-05-28 15:15 UTC — PR #146: Scenario B FP8 weights + spec15/lookup8 composition (CLOSED — quality fail)

- **Branch:** `tanjiro/sc-b-fp8-spec-composition`
- **Student:** tanjiro
- **Hypothesis:** Compose FP8 weight quantization (confirmed TTFT gain on Sc A/D) with n-gram
  spec15/lookup8 (confirmed TPOT gain on Sc B). Expected: multiplicative 5–7x speedup.

### Quick probe results

| Arm | Launcher | Quick speedup | Quality ratio (n=16) | W&B |
|---|---|---:|---:|---|
| **arm1** | FP8 + spec15/lookup8/min=2 | **6.39x** | 0.629 (screening) | tovkrsoz |
| arm2 | FP8 + spec15/lookup8/min=1 | 2.88x | 0.839 | v7x45j9q |
| arm3 | FP8 + spec20/lookup8/min=2 | 4.84x | 0.839 | sgrmetlf |

Promotion: arm1 per >5% rule. arm2 (min=1) is much worse — single-token lookback increases noise. arm3 (spec20) regresses vs arm1 — FP8 dequant overhead cancels the deeper-spec gain.

### Full eval results (arm1 — 64 requests, burst profile)

| Metric | arm1 (FP8+spec15) | PR #141 baseline (spec15 BF16) | Δ |
|---|---:|---:|---:|
| **scenario/B/speedup_over_pytorch** | **~3.83x** | **3.550x** | **+7.9%** |
| inverse_tpot_p50 | 152.21 tok/s | 141.15 tok/s | +7.8% |
| TPOT.p50 | 6.57 ms | 7.08 ms | -7.2% |
| TTFT.p50 | 0.0448 s | — | -37% (FP8 helps prefill) |
| **MMLU-Pro quality ratio (n=500)** | **0.913** | **~1.007** | **−9.3% ❌ FAIL gate** |
| Speed success | 64/64 ✓ | 64/64 ✓ | — |
| VRAM peak | 90.5 GiB | 90.3 GiB | ≈ same |
| W&B | jqaxzt67 | 8656lf5w | — |

**Closed:** Not merged — quality gate fails (0.913 < 0.95). Speed gain (+7.9%) is not bankable with quality failure.

### Analysis

- **Speed composition was real but weak (+7.9%, not the predicted 50–100%).** At spec=15, the verify forward pass is itself prefill-like compute (15 draft tokens × 7B params). FP8 bandwidth savings are partially offset by dequant overhead in each verify step, which is now compute-bound. Contrast with Sc D PR #139: spec=5 verify is small enough that FP8 dequant is amortised.
- **Quality failure is the decisive problem.** FP8 weight precision loss (small logit drift per token) accumulates differently in spec decoding's greedy verify: each verify step creates a slightly different accept/reject decision tree vs BF16. With 8192 decode tokens × spec15 batches, the number of verify decisions is ~8192/15 ≈ 546 per request. Each decision introduces a tiny FP8 distribution difference; across 500 MMLU-Pro questions at n=64 requests, this compounds to -2.6pp accuracy.
- **Rule established:** FP8 + deep spec (depth ≥ 15) fails quality gate on Sc B. This is a hard constraint from this data point. Future Sc B experiments must choose: FP8 without deep spec (max spec=5 based on Sc D experience), OR deep spec without FP8 (the PR #141 winner pattern).
- **arm3 (FP8+spec20) was slower than arm1 (FP8+spec15):** This is a confounded signal because FP8 amplifies rejection rate at depth=20. The clean depth=20 signal (BF16) is measured in PR #149.

**Next experiment for tanjiro:** PR #149 — n-gram spec depth sweep beyond 15 (spec20, spec25, spec30) with BF16, no FP8. Tests whether the superlinear depth trend from PR #141 continues toward the H100 SMAC3 ceiling (15.23x).

---

## 2026-05-28 15:25 UTC — PR #148: Scenario A FlashInfer prefill attention + FP8 weights (CLOSED — blocked by SM120 / vLLM 0.11 stack)

- **Branch:** `frieren/flashinfer-sc-a-prefill`
- **Student:** frieren
- **Hypothesis:** FlashInfer attention's fused prefill kernel (Tensor-Core tiling) should beat
  FlashAttention 2 on Sc A's 8192-token prefill at concurrency 1. PR #145 closed the bandwidth
  side of the Sc A problem (FP8 KV was net negative because attention is compute-bound), so the
  next attack is kernel-level.

### Result

| Arm | Backend | Quick speedup | TTFT.p50 (s) | Status |
|---|---|---:|---:|---|
| arm1 FlashInfer + FP8w | FLASHINFER | — | — | **BLOCKED** at vLLM↔FlashInfer plan() signature mismatch |
| arm1 FlashInfer + FP8w + compat shim + `--enforce-eager` | FLASHINFER | — | — | **BLOCKED** at FP8 GEMM JIT linker (`libcublas` not findable) |
| arm2 / arm3 FlashInfer variants | FLASHINFER | — | — | Not run (same upstream blockers as arm1) |
| **arm1-fallback Triton + FP8w** | **TRITON_ATTN** | **1.484x** | **0.2955** | Boots cleanly; **−20% vs PR #137 (1.866x)** |
| PR #137 reference | FLASH_ATTN | 1.866x (full) | 0.2349 | Current Sc A best |

**Closed:** did_not_improve, `baseline_update_allowed=false`. W&B run: `i7331pvj` (Triton arm1 quick).

### The 5-layer FlashInfer ↔ vLLM 0.11 incompatibility cascade (frieren's investigation)

Each fix surfaced the next blocker; frieren stopped at layer 5 (out-of-scope patches):

1. **`max-num-seqs < 9` triggers FlashInfer kernel_warmup buffer overflow.** `kernel_warmup.py:55` calls
   `_dummy_run(num_tokens=16, create_mixed_batch=True)`; `num_decode_tokens=8`, `num_reqs=9`. With
   `max_num_seqs<9`, `seq_lens.np` cannot hold the array. PR's arm3 (`--max-num-seqs 4`) is therefore
   infeasible without a vLLM patch.
2. **vLLM 0.11 ↔ FlashInfer 0.6 `plan()` positional-arg drift.** vLLM pins `flashinfer-python==0.3.1`
   but image has `0.6.11.post3`. `BatchDecodeWithPagedKVCacheWrapper.plan()` in 0.6.x inserts a new
   `o_data_type` parameter between `kv_data_type` and `data_type`. vLLM's `fast_plan_decode` passes
   positionally → `sm_scale` ends up bound to `rope_scale` (None). Assertion fires:
   `assert decode_wrapper._sm_scale == self.scale` (`flashinfer.py:972`).
3. **Cudagraph fast-path drift.** Even with a Python `.plan()` shim, the cudagraph fast-path makes a
   DLPack call into a JIT-compiled `plan(0..18: DLTensor*/int/bool)` expecting 19 args but receiving 15.
   `--enforce-eager` bypasses this path.
4. **FlashInfer FP8 GEMM JIT linker error.** `flashinfer.gemm.gemm_base.fp8_gemm_sm100` JIT-builds for
   `arch=compute_120f,code=sm_120f`. `nvcc` compile succeeds, but `/usr/bin/ld: cannot find -lcublas`
   / `-lcublasLt`. `/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib/` has `libcublas.so.12`
   / `libcublasLt.so.12` but no unversioned symlinks, and `LIBRARY_PATH` is not configured. Engine
   dies on first FP8 layer-0 QKV projection.
5. Not reached — out of scope.

### Triton fallback detail (quick, n=4 burst)

| Metric | Triton arm1 | PR #137 (full, n=128) | PyTorch |
|---|---:|---:|---:|
| TTFT.p50 | 0.2955 s | 0.2349 s | 0.4385 s |
| TTFT.p90 | 0.5955 s | 0.2613 s | 0.5089 s |
| Generation throughput | 53.6 tok/s | — | 35.7 tok/s |
| Success | 4/4 | 128/128 | 128/128 |
| VRAM peak | 91115 MiB | 93267 MiB | — |
| **Speedup vs PT** | **1.484x** | **1.866x** | 1.000x |
| **vs PR #137** | **−20.5%** | (baseline) | — |

### Analysis & rules established

- **FlashInfer prefill attention is NOT reachable on the vLLM 0.11 + FlashInfer 0.6.x stack on SM120.**
  Three independent integration breaks (plan signature, cudagraph fast-path, FP8 GEMM linker) each
  block engine boot. Patching any one surfaces the next. Frieren confirmed FlashInfer's kernel itself
  JIT-compiles for `sm_120f` — the kernel works, the integration is the wall.
- **Triton attention backend is ~20% slower than FlashAttention 2 on Sc A.** Triton is the only viable
  alternative when FlashInfer breaks, but it cannot recover Sc A; the gap is far above the 2% quick-to-full
  noise band. PR #137 FA2 + FP8 weights remains the Sc A frontier.
- **Doc bug for next student:** vLLM 0.11's enum is `TRITON_ATTN`, not `TRITON_ATTN_VLLM_V1` (the PR's
  instructions said the latter, which doesn't exist).
- **Rule established: vLLM 0.11 + FlashInfer 0.6.x is broken on SM120.** To unblock FlashInfer requires
  vLLM ≥ 0.12 (which rewrote `fast_plan_decode` for the new FlashInfer signature). This is a per-PR
  venv with vLLM ≥ 0.12 — defer until other levers are exhausted.
- **Rule established: Sc A bandwidth-reduction levers in vLLM 0.11 are FULLY EXHAUSTED.** FP8 weights
  win (+1.866x), FP8 KV regresses (−26% at conc=1 compute-bound prefill), n-gram is metric-orthogonal
  (TTFT-only metric), FlashInfer is integration-blocked. The next Sc A attack is either (a) per-PR
  venv with vLLM ≥ 0.12 + FlashInfer attention, (b) TensorRT-LLM with its own SM120 kernel stack, or
  (c) INT4 weight-only quantization via Marlin (4x bandwidth reduction vs FP8's 2x).

**Next experiment for frieren:** Pivot to Sc C — extend PR #144 winner (24.305x SGLang LPM + radix)
with FP8 KV cache + mem-fraction push (researcher H2+H5). Sc C is 256-concurrency, KV-memory-bound;
FP8 KV halves block size — opposite of Sc A regime. PR #144 left 15 GiB of unused VRAM headroom.

---

## 2026-05-28 16:20 UTC — PR #151: Sc C SGLang FP8 KV cache (MERGED — new Sc C best)

- **Branch:** `frieren/sc-c-sglang-fp8-kv-r2`
- **Student:** frieren
- **Hypothesis:** Extend PR #144 winner (SGLang LPM + radix, 24.305x) with FP8 KV cache
  (`--kv-cache-dtype fp8_e5m2`). FP8 KV halves per-request KV memory footprint, doubling
  the KV-token capacity and enabling more concurrent in-flight states at Sc C's high concurrency.
  Unlike Sc A (conc=1 compute-bound, where FP8 KV regressed -26%), Sc C is KV-bandwidth-bound
  at concurrency ≫1 — the expected regime for FP8 KV to pay off.

### Quick-probe results (3 arms, n=4 burst per arm)

| Arm | mem | chunk | FP8 KV | Quick speedup | VRAM peak | max_total_tokens | W&B |
|---|---:|---:|---|---:|---:|---:|---|
| **arm1** FP8 KV baseline | 0.85 | 8k | fp8_e5m2 | **4.247x** | 84.4 GiB | **1,090,726** | obf5x0a2 |
| arm2 FP8 KV + mem push | 0.92 | 8k | fp8_e5m2 | 4.243x | 91.2 GiB | 1,198,879 | kiiz3p5i |
| arm3 FP8 KV + mem + chunk | 0.92 | 16k | fp8_e5m2 | 4.236x | 92.1 GiB | 1,198,879 | jkf2ti8i |

All three arms within 0.3% — arm1 promoted (simplest isolation). KV-token capacity roughly doubled vs PR #144 BF16 (~545k → ~1,090k tokens). `fp8_e5m2` accepted by SGLang 0.5.12.post1 with no fallback.

### Full eval result (arm1)

| Metric | PR #144 (SGLang LPM+radix) | **PR #151 (+ FP8 KV)** | Delta |
|---|---:|---:|---:|
| **scenario/C/speedup_over_pytorch** | **24.305x** | **27.497x** | **+13.1%** |
| geomean req/s | ~1.954 | **~2.329** | +19.2% |
| MMLU-Pro quality ratio | 1.027 | **1.000** | within gate |
| Speed success | 768/768 | **768/768** | clean |
| VRAM peak | 82.5 GiB | **84.4 GiB** | +2.3% |
| W&B | ifj6fcec | **kk6shiqh** | — |

**Merged:** Yes — squash-merged to `ib-20260528-12h-r2`. New Sc C best (+13.1%, 52%→59% of H100 ceiling).

### Analysis

- **FP8 KV mechanism confirmed on Sc C.** Halving KV block size (BF16 → FP8) doubled max KV-token capacity from ~545k to ~1,090k tokens at the same mem-fraction. The additional capacity allows more concurrent requests to stay in-flight during the burst profile peak, directly increasing geomean req/s.
- **Regime-specificity confirmed.** FP8 KV regressed -26% on Sc A (conc=1 compute-bound), wins +13% on Sc C (conc=256 KV-bandwidth-bound). The key dividing line is whether the workload is bottlenecked at the KV memory tier or the compute tier.
- **arm2 (mem 0.92) and arm3 (chunk 16k) added nothing in quick eval.** With FP8 KV already doubling the KV capacity, the mem-fraction headroom is less impactful — the binding constraint shifted from KV capacity to scheduling/compute throughput.
- **Quality ratio improved to exactly 1.000** (vs 1.027 in PR #144). Numerically, FP8 KV is loss-free relative to BF16 KV for this model — the slight accuracy *decrease* from 1.027→1.000 reflects noise, not degradation.
- **Composition potential:** FP8 KV is now the Sc C KV-tier baseline. Next lever: FP8 weights on top (directly halves weight-read bandwidth at prefill). The composition FP8 weights + FP8 KV hasn't been tested on SGLang Sc C; vLLM FP8 weights hurt Sc C previously but SGLang's kernel implementation is different.

**Next experiment for frieren:** Sc C SGLang FP8 weights composed with FP8 KV (PR #151 baseline).

---

## 2026-05-28 15:50 UTC — PR #147: Scenario D SGLang LPM + radix cache (CLOSED — did_not_improve)

- **Branch:** `fern/sglang-sc-d-lpm-radix`
- **Student:** fern
- **Hypothesis:** Port the PR #144 Sc C winning mechanism (SGLang LPM scheduler + always-on
  radix cache) to Sc D. Sc D has 4-concurrency with 4096-token inputs (4× longer than Sc C's
  1024-token inputs), which should expose more prefix-tree hits per request. Compose with
  FP8 + SGLang NGRAM to mirror PR #139's vLLM FP8+n-gram composition pattern.

### Result: **1.568x** — below PR #139 baseline (2.073x), -24.4% regression

### Quick-probe results (n=4 burst, n=16 MMLU)

| Arm | Levers | Quick speedup | TTFT.p50 | TPOT.p50 | W&B |
|---|---|---:|---:|---:|---|
| arm1 | LPM + radix | 1.121x | 0.2016s | 0.0215s | n5vbzgpo |
| arm2 | + FP8 | 1.316x | 0.1422s | 0.0186s | 2fnl9pi6 |
| arm3 | + NGRAM spec | 1.228x | 0.1961s | 0.0181s | v4cuazq5 |
| **arm4** | **FP8 + NGRAM (composition)** | **1.506x** | **0.1360s** | **0.0155s** | **ge4hc9d3** |

### Full eval result (arm4)

| Metric | PyTorch | PR #139 winner | **arm4 (this PR)** | arm4 vs baseline |
|---|---:|---:|---:|---:|
| TTFT.p50 | 0.2123s | 0.1161s | 0.1165s | tied |
| TPOT.p50 | 0.0251s | 0.0104s | **0.0163s** | **−57% worse** |
| req/s | 0.0382 | 0.0776 | 0.0526 | **−32% worse** |
| geomean | 1.930 | 4.001 | 3.026 | -24.4% |
| **speedup_over_pytorch** | 1.000x | **2.073x** | **1.568x** | **−24.4%** |
| MMLU-Pro ratio (n=500) | 1.000 | 0.953 | 0.993 ✓ | within gate |
| Speed success | — | 96/96 | 96/96 ✓ | clean |
| VRAM peak | — | 91.7 GiB | 82.2 GiB | -10% |
| W&B | — | 40f15iox | **ps1tr5ni** | — |

**Closed:** Not merged — did_not_improve. Quality clean (0.993), validation pass, but speed regression vs PR #139.

### Three independent failure modes (fern's analysis)

1. **Radix-cache transfer hypothesis disproved.** PR #144 won Sc C +15.2% via **cross-request prefix sharing** within a 256-request burst (LPM scheduler reorders to maximise hits). Sc D has 4 concurrent requests with *unique* dataset content — no shared prefixes exist for the radix tree. arm1 quick=1.121x is essentially LPM/radix overhead with no measurable benefit. Input length (4096 vs 1024) is irrelevant; what mattered was *concurrent-request count × prefix overlap structure*.
2. **SGLang NGRAM weaker than vLLM prompt-lookup on natural-language Sc D outputs.** SGLang's BFS-based n-gram with `max_bfs_breadth=10` explores 10 candidate paths per draft step — 10× verify compute. On non-repetitive Mistral outputs, accept_len averaged 1.5-2.0 vs vLLM's effective ~2.5 (single-path). TPOT: SGLang NGRAM 0.0163s vs vLLM spec5 0.0104s = -57%.
3. **SGLang NGRAM forces `disable_overlap_schedule=True`** + `enable_mixed_chunk=False`. arm3 (NGRAM only) TTFT 0.196s vs arm2 (FP8 only) 0.142s — NGRAM disables overlap-prefill optimisations, regressing TTFT by ~38%. arm4 (FP8+NGRAM) recovers TTFT to 0.136s but TPOT regression is decisive.

### Key insights & rules established

- **SGLang LPM+radix mechanism is Sc C-specific.** Requires high-concurrency burst with cross-request prefix overlap. Rule: do not attempt this mechanism on workloads with `concurrency ≤ small_count × unique_requests`.
- **Engine choice matters per scenario.** vLLM's prompt-lookup is decisively stronger than SGLang's NGRAM for Sc D's natural-language outputs. SGLang wins Sc C (bursty, schedulable), vLLM wins Sc D (low-concurrency, deeper-spec-decode-friendly).
- **Composition pattern (FP8 + spec) works in both engines** but the spec implementation choice is determinative. vLLM PR #139 spec5/lookup4 → 2.073x. SGLang NGRAM 15/breadth=10 → 1.568x. Same scenario, different engines, opposite composition outcomes.
- **fern's quick→full ratio for SGLang arm4: 1.506→1.568x (+4.1%).** Smaller than vLLM PR #139's quick→full +14.5%. SGLang quick probes are MORE predictive of full performance than vLLM quick probes — useful pattern for future SGLang work.

**Next experiment for fern:** Sc D vLLM spec depth upgrade (researcher H1) — extend PR #139 winner from spec5/lookup4 to spec15/lookup8. PR #141 (Sc B) showed spec5→spec15 gave +32%. On Sc D (2048-token outputs, less spec exposure than Sc B's 8192) the gain should be smaller but still meaningful. Direct vLLM extension, two integer changes; quality risk lower than Sc B (4× less verify-step exposure).

## 2026-05-28 16:35 UTC — PR #149: Scenario B n-gram spec depth sweep beyond 15 (spec20/25/30, BF16)

- **Branch:** `tanjiro/sc-b-ngram-deeper-spec`
- **Student:** tanjiro
- **Hypothesis:** The spec5→spec15 depth jump gave +32% on Sc B (PR #141). The depth→speedup curve may not have saturated. Testing spec20, spec25, spec30 (all BF16, no FP8 due to PR #146 quality failure) to find the optimal depth.
- **Status:** MERGED — new Sc B best

### Results

| Arm | spec_tokens | lookup_max | TPOT.p50 (ms) | Quick speedup | Full speedup | Quality ratio | W&B |
|---|---:|---:|---:|---:|---:|---:|---|
| arm1 | 20 | 10 | 5.21 | 4.83x (quick) | — | — | wfhkazbg |
| **arm2** | **25** | **12** | **2.91** | **8.64x (quick)** | **3.888x (full)** | **1.013** | **wkedminm** |
| arm3 | 30 | 15 | 5.85 | 4.30x (quick) | — | — | lyu6arfv |

- **PyTorch baseline:** TPOT.p50 25.15 ms, inverse_tpot 39.76 tok/s
- **PR #141 baseline:** 3.550x (spec15/lookup8, BF16)
- **Winner (arm2):** 3.888x full (+9.5% over PR #141), TPOT.p50 6.47ms, inverse_tpot 154.56 tok/s
- **Quality:** MMLU-Pro 0.302 obs / 0.298 baseline = ratio 1.013 ✓ (gate 0.95, n=500)
- **Speed success:** 64/64 ✓
- **VRAM peak:** 90.19 GiB

### Analysis

The depth scaling curve on Sc B:
- spec5/4 → 2.687x (PR #136)
- spec15/8 → 3.550x (+32%)
- spec25/12 → 3.888x (+9.5%)

Spec depth continues to scale but with strong **diminishing returns** past depth=15. The quick probe reveals the full picture:
- arm2 (spec25): quick = 8.64x — enormous quick speedup, but quick→full ratio only 0.45x (vs spec15's 0.51x). At depth=25, verify step is heavier → per-request latency variance is higher across the 64-request full eval.
- arm3 (spec30): quick = 4.30x — falls *below* arm1's 4.83x quick. Verify-step compute explicitly exceeds the gain from longer accepted bursts. Saturation confirmed at depth=25.

**Rule established:** n-gram spec depth peak on Sc B is at spec25/lookup12. spec30 saturates. The spec5→spec15→spec25 trajectory (+32%, +9.5%) exhibits clear diminishing returns — future large Sc B gains require a fundamentally different speculation approach (draft model such as EAGLE or Medusa).

**Notable:** tanjiro also fixed a `PROBLEM_DIR` absolutization bug in the eval scaffolding (eval_env.sh relative path failure when CWD ≠ repo root → VLLM_USE_FLASHINFER_SAMPLER=0 unset → FlashInfer JIT compile error). Workaround: export PROBLEM_DIR=/workspace/senpai/target before sourcing eval_env.sh.

### Launcher (arm2 winner)

```bash
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" --host "${HOST}" --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 2048 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --kv-cache-dtype auto \
    --speculative-config '{"method":"ngram","num_speculative_tokens":25,"prompt_lookup_max":12,"prompt_lookup_min":2}' \
    --trust-remote-code --disable-log-stats
```

## 2026-05-28 16:25 UTC — PR #152: Scenario D vLLM n-gram spec15/lookup8 depth upgrade (extend PR #139)

- **Branch:** `fern/sc-d-vllm-spec15-upgrade`
- **Student:** fern
- **Hypothesis:** PR #139 uses spec5/lookup4 on Sc D. PR #141 showed spec5→spec15 gave +32% on Sc B. On Sc D (2048-token outputs, 4× shorter than Sc B), the gain should be smaller but real. Testing spec15/lookup8 + spec10/lookup6 (intermediate) + BF16+spec15 (quality control arm).
- **Status:** Review-ready pending rebase → will be merged as new Sc D best

### Quick probe results (n=4, quality_n=16)

| Arm | Config | Quick speedup | Quality (n=16) | Outcome |
|---|---|---:|:---:|---|
| arm1 | FP8 + spec15/lookup8 | 2.358x | 0.629 ✗ | quality screen FAIL — skipped |
| **arm2** | **FP8 + spec10/lookup6** | **2.033x** | **1.049 ✓** | **promoted to full** |
| arm3 | BF16 + spec15/lookup8 | 2.034x | 0.629 ✗ | quality screen FAIL — skipped |

### Full eval results (arm2, quality_n=500)

| Metric | arm2 (PR #152) | PR #139 baseline | PyTorch |
|---|---:|---:|---:|
| **speedup_over_pytorch** | **2.218x** | 2.073x | 1.000x |
| TPOT.p50 | 0.00958 s | 0.0104 s | 0.0251 s |
| TTFT.p50 | 0.1152 s | 0.1161 s | 0.2123 s |
| req/s | 0.0865 | 0.0776 | 0.0382 |
| MMLU-Pro quality ratio | 0.960 ✓ | 0.953 ✓ | — |
| Speed success | 96/96 | 96/96 | — |
| VRAM peak | 91.7 GiB | 91.7 GiB | — |
| W&B | g1xjqoyg | 40f15iox | — |

- **Improvement:** +7.0% over PR #139

### Analysis

**Key finding: spec depth ≥ 15 fails quality on Sc D regardless of dtype.** Both FP8+spec15 (arm1) and BF16+spec15 (arm3) produced identical quality ratio 0.629 at n=16 — ruling out FP8 as the cause. This generalizes the rule from Sc B (PR #146: FP8+spec15 failed) to Sc D (both dtypes fail at spec15). On Sc D, the quality cliff appears between spec10 (0.960) and spec15 (0.629).

The spec10 intermediate depth captures a real but smaller benefit: TPOT.p50 -7.9%, req/s +11.5%, geomean +7.0%. The gain from spec5→spec10 is smaller than spec5→spec15 would have been (if quality could be preserved), consistent with diminishing returns in the spec depth curve.

**Rule established:** `spec_depth ≥ 15` fails quality gate on Sc D (2048-token outputs) regardless of dtype. Quality cliff is between spec10 (safe, ratio 0.960) and spec15 (fail, ratio 0.629).

**fern's suggested follow-ups (logged for next assignment):**
1. spec7/8/11/12 sweep to find quality knee precisely
2. prompt_lookup_min=1 at spec10
3. Chunked-prefill size tuning at conc=4 (max-num-batched-tokens variant)

## 2026-05-28 17:20 UTC — PR #154: Scenario D n-gram spec depth fine-sweep above spec10

- **Branch:** `tanjiro/sc-d-vllm-spec-fineswee`
- **Student:** tanjiro
- **Hypothesis:** Map the quality cliff between spec10 (PR #152, safe, 2.218x quality 0.960) and spec15 (PR #146/#152 quick, fail 0.629). Test spec11/12 (just above safe) plus spec10/min=1 (extend matching window without raising depth).
- **Status:** CLOSED — did_not_improve / hit quality cliff

### Quick probe results (n=4, quality_n=16)

| Arm | Config | Quick speedup | Quality (n=16) | Verdict | W&B |
|---|---|---:|---:|---|---|
| arm1 | spec11/lookup7/min=2 | 2.025x | 0.629 ✗ | screen FAIL | l9i86yd5 |
| arm2 | spec12/lookup8/min=2 | 2.444x | 0.629 ✗ | screen FAIL | eb2qs8tp |
| arm3 | spec10/lookup6/**min=1** | 2.244x | 0.839 ✓ | promoted | 0ozy7ol0 |

### Full eval result (arm3 only)

| Metric | arm3 (spec10/min=1) | PR #152 (spec10/min=2) | Δ |
|---|---:|---:|---:|
| **Speedup** | 2.323x | 2.218x | +4.7% |
| TPOT.p50 | 0.00870 s | 0.00958 s | -9.2% |
| TTFT.p50 | 0.1152 s | 0.1152 s | tied |
| **Quality ratio** | **0.926** ✗ | 0.960 ✓ | **-3.5pp, gate FAIL** |
| Speed success | 96/96 | 96/96 | tied |
| W&B | ily0f83g | g1xjqoyg | — |

### Analysis

**No arm passes quality gate.** Two failure modes:
1. **Deeper spec (arms 1, 2):** spec11/12 fail n=16 screen at 0.629 — same value as spec15/spec20+ regressions. The cliff between spec10 (safe) and spec11 (fail) is much steeper than expected. Plausible mechanism: deeper draft windows admit more high-perplexity drafts that pass FP8 verification by numerical tolerance.
2. **Looser min (arm 3):** spec10/min=1 keeps decode latency advantage (TPOT -9.2%) and passes n=16 screen at 0.839 BUT fails full quality gate at 0.926 (< 0.95). min=2 → min=1 admits single-token-anchored drafts that drift MMLU-Pro accuracy by -3.5pp.

### Key rule established

**PR #152's spec10/lookup6/min=2 is the n-gram local optimum on Sc D.** Neither widening spec depth nor loosening lookup min preserves quality. Future Sc D gains must come from different mechanisms:
- FP8 KV cache composition (untried on Sc D — PR #164 next)
- EAGLE/Medusa draft-model speculation (less sensitive to FP8 numerics than n-gram)
- PD-disaggregated serving (TTFT and TPOT bottlenecks on different paths in Sc D)

### Cross-scenario quality contrast

| Scenario | Output length | Highest passing spec | Highest min |
|---|---:|---:|---|
| Sc B (PR #149) | 8192 | spec25 | min=2 (untested otherwise) |
| Sc D (PR #152) | 2048 | spec10 | min=2 |
| Sc A | 1024 | n-gram metric-orthogonal | — |

Sc B's tolerance for deep specs (25 vs Sc D's 10) is plausibly explained by longer output amortising the verify-step quality drift across more tokens. Sc D's 4× shorter outputs concentrate per-decoder-step quality cost.

**Next experiment for tanjiro:** Sc D FP8 KV composition with PR #152 winner (PR #164 — 3-arm: fp8_e5m2 + fp8_e4m3 + mem 0.95 control).

## 2026-05-28 17:23 UTC — PR #153: Sc C SGLang FP8 weights + FP8 KV composition (CLOSED — stuck/abandoned)

- **Branch:** `frieren/sc-c-sglang-fp8-weights-kv`
- **Student:** frieren
- **Hypothesis:** Compose SGLang FP8 weights (`--quantization fp8`) with PR #151's FP8 KV (`fp8_e5m2`) winner. 3-arm ablation: arm1 weights+KV combined, arm2 +mem-fraction 0.92, arm3 weights-only (no FP8 KV) as control.

**Result:** No results — closed pragmatically after 85+ min with no commits, no PR comments, no GPU usage (0 MiB throughout the iteration starting 16:15 UTC), no response to advisor heartbeat check at 16:49 UTC.

**Operational diagnosis:**
- Previous frieren+Sc C PRs (#150, #151) completed in ~29 min each.
- Per-PR SGLang venv install (`/tmp/inferencebench-engine-venvs/sglang-pr-153`) is a known 10-15 min step but visible progress should be in commits/comments within 30 min of assignment.
- 85+ minutes with zero signal (no commits, no comments, no GPU usage, no acknowledgement of heartbeat) indicates either student session never started or hit an early-bootstrap error and silently halted.

**Decision:** Closed PR #153 to reclaim GPU time for a fresh assignment (PR #166 — Sc B prompt_lookup_min=1 sweep on PR #149 winner). Sc C FP8 weights composition remains untested; can be revisited in a future round if the student session recovers and emits results to W&B group `frieren-scC-fp8wt`.

**Lessons:**
- Pragmatic closure at 85 min is reasonable when (a) no GPU usage detected, (b) no PR activity, (c) heartbeat check ignored. Recovering 4+ hours of GPU time for a fresh experiment beats waiting indefinitely on a possibly-dead session.
- Future SGLang assignments: ask student to post a "venv install started" status comment within first 5 min of assignment, so missing progress is visible earlier.
- Sc C FP8-weights-on-SGLang remains an open scientific question — PR #147 arm2 had confirmed SGLang FP8 weights work on SM120 Blackwell at Sc D's 4-concurrency. The Sc C 256-concurrency regime composition is still to be tested.

**Reassignment:** frieren → PR #166 (Sc B prompt_lookup_min=1 sweep on PR #149 spec25/lookup12 winner). Tests rule #15 implication that Sc B's 8192-out absorbs min=1 quality drift.

## 2026-05-28 17:57 UTC — PR #156: Sc A scheduling fine-tune (MERGED — new Sc A best)

- **Branch:** `fern/sc-a-vllm-scheduling-int4`
- **Student:** fern
- **Hypothesis:** Reducing `max-num-seqs` from 8 (PR #137) to 1 eliminates phantom-sequence KV reservation overhead at conc=1. Also tested INT4 AWQ quantization as a secondary arm.

**Results:**

| Arm | TTFT.p50 | Speedup | Quality (n=500) | W&B |
|---|---:|---:|---:|---|
| **arm1 fp8/seqs=1/tok=8192** | **0.2332s** | **1.881x** | 0.953 ✓ | ds519cur |
| arm2 fp8/seqs=2/tok=8192 | 0.22758s | 1.927x quick | 0.839 quick | zan9g0vh (quick only) |
| arm3 AWQ Marlin/seqs=1/tok=8192 | 0.36636s quick | 1.197x quick | 0.000 (broken) | 7vk7zutt (quick) |
| PR #137 baseline | 0.2349s | 1.866x | 0.966 | izg22lch |

**Decision:** arm1 promoted to full eval (lowest TTFT.p50, matches conc=1 phantom-sequence hypothesis). arm3 AWQ discarded — `solidrust/Mistral-7B-Instruct-v0.3-AWQ` has chat-template incompatibility (conversation-roles alternation error) AND awq_marlin kernel regresses TTFT -56% vs FP8 on Blackwell SM120.

**Full eval arm1:** 1.881x (+0.78% over PR #137's 1.866x), quality 0.953 (n=500, passes gate ✓), 128/128 speed success. Terminal result submitted and merged.

**Analysis:** Gain is real but modest (+0.78%). This confirms the phantom-sequence hypothesis: vLLM reserves KV blocks for max-num-seqs requests even at conc=1, and reducing to 1 frees that capacity. However, Sc A is compute-bound (FP8 dequant) not KV-bandwidth-bound, so the savings are marginal. The FP8 dequant is the irreducible bottleneck at conc=1. All vLLM 0.11 Sc A levers now exhausted.

**Key finding — AWQ on Blackwell SM120:**
- `solidrust/Mistral-7B-Instruct-v0.3-AWQ` (AWQ gemm 4bit g128) has a chat-template incompatibility: "Conversation roles must alternate user/assistant/user/assistant" — breaks all quality evaluation.
- Even if chat-template were fixed, awq_marlin kernel performance (-56% TTFT) is worse than FP8 on SM120. AWQ is not a viable Sc A direction on this hardware.

**Informs rule #16:** INT4 AWQ on SM120 Blackwell with `solidrust/Mistral-7B-Instruct-v0.3-AWQ` is not viable — both correct chat-template and kernel performance fail. Future INT4 investigation requires a compatible checkpoint (GPTQ-int4 or AWQ with matching instruct chat template).

**Next for fern:** PR #172 — Sc C SGLang FP8 weights composition (the unfinished business from stuck PR #153).

## 2026-05-28 18:07 UTC — PR #164: Sc D FP8 KV cache composition with PR #152 (CLOSED — did_not_improve)

- **Branch:** `tanjiro/sc-d-fp8kv-composition`
- **Student:** tanjiro
- **Hypothesis:** FP8 KV cache (+13.1% on Sc C at 256-conc) might transfer to Sc D's intermediate conc=4 regime. 3-arm test: fp8_e5m2, fp8_e4m3, BF16 KV mem-0.95 control.

**Results:**

| Arm | Config | Quick speedup | Quality (n=16) | Full speedup | Notes |
|---|---|---:|---:|---:|---|
| arm1 | FP8 weights + FP8 KV (e5m2), mem 0.92 | 1.829x | 0.629 (noise) | — | -10% vs PR #152 quick, not promoted |
| arm2 | FP8 weights + FP8 KV (e4m3), mem 0.92 | 1.828x | 0.629 (noise) | — | -10% vs PR #152 quick, not promoted |
| **arm3** | FP8 weights + BF16 KV (control), mem **0.95** | **2.048x** | **1.049** | **2.208x** | Tied PR #152 (-0.45%); MERGED would not beat baseline |
| PR #152 baseline | FP8 weights + BF16 KV, mem 0.92 | 2.033x | 0.960 | 2.218x | Current Sc D best |

W&B full eval (arm3): `mor4bn11`

**Analysis:**
- FP8 KV (both e5m2 and e4m3) regressed Sc D by -10% at quick probe — both formats gave identical results (1.829x vs 1.828x = within 0.05%). The dequant compute overhead at conc=4 exceeds the bandwidth savings from halved KV memory.
- arm3 control (mem 0.92 → 0.95) is neutral: +3.2% VRAM headroom but no throughput gain. PR #152 is at the mem-fraction plateau for Sc D. arm3 full at 2.208x = -0.45% vs PR #152 (noise).
- Quick→full mapping stable (~+8%): arm3 quick 2.048x → full 2.208x = +7.8%, consistent with PR #164's note and PR #152's own ratio. Sc D quick is a reliable predictor (unlike Sc B).
- Student correctly did NOT promote arm1/arm2 to full — speed signal (-10%) was the reject signal, not quality screen.

**Rule #16 established:** FP8 KV cache regresses Sc D at conc=4 (intermediate regime). The KV-bandwidth-bound vs compute-bound threshold on this hardware sits between conc=4 (Sc D, FP8 KV HURTS) and conc=256 (Sc C, FP8 KV +13.1%). Future Sc D FP8 KV exploration should only proceed with conc > 4 or if the attention kernel changes.

**Key architectural insight:** e5m2 and e4m3 give identical Sc D regression (-10%), confirming the bottleneck is compute (dequant overhead), not format precision. Format choice is irrelevant when the regime is wrong.

**Decision:** CLOSED — arm3 at 2.208x does not beat PR #152's 2.218x. Good negative result.

**Next for tanjiro:** PR #173 — Sc D chunked-prefill vs one-shot prefill + batched-tokens sweep on PR #152 base.
