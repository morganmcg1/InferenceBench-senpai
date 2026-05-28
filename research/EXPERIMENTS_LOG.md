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
