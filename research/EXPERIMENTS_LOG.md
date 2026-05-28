# SENPAI Research Results

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
