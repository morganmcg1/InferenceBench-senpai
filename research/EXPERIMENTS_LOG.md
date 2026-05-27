# SENPAI Research Results — ib-20260527-lean1-r1

## 2026-05-27 16:46 — PR #128: Scenario C — vLLM FP8 + large batch + prefix caching

- Branch: `tanjiro/scenario-c-throughput-fp8-large-batch`
- Student: tanjiro
- Status: **MERGED — first measured Scenario C row in BASELINE.md**

### Hypothesis

Scenario C is high-load throughput (256 requests across 3 traffic profiles: `burst` c=64, `poisson` 32 r/s, `constant` 16 r/s). Test whether vLLM with `--max-num-seqs=256`, `--max-num-batched-tokens=16384`, `--enable-chunked-prefill`, `--enable-prefix-caching`, `--quantization fp8`, and `--gpu-memory-utilization=0.92` beats the implicit PyTorch baseline.

### Results

| Run | Eval mode | Primary metric | Value | Quality ratio | Quality pass | terminal_eligible |
|---|---|---|---:|---:|---|---|
| `ocz25mgd` | quick | scenario/C/speedup_over_pytorch (geomean) | 2.76x | 0.839 (n=16 screen) | screening-only | False |
| `ckfmuinz` | full | scenario/C/speedup_over_pytorch (geomean) | **25.62x** | **1.0** (n=500) | True | **True** |

- Speed requests: 768/768 succeeded across all 3 profiles, 0 failures.
- Validation: `validation_pass=true`, `baseline_update_allowed=true`.
- VRAM peak: ~89.7 GiB on RTX PRO 6000 (~96 GiB available).
- Raw geomean throughput: 2.17 r/s vs PyTorch baseline 0.0847 r/s.

### Analysis

- The 25.62x speedup is the **first measured Scenario C baseline on RTX PRO 6000** and substantially above the PyTorch baseline.
- Reference (H100 paper, not measured): vLLM default 48.69x. Our 25.62x is below that, consistent with RTX PRO 6000 being slower per-SM than H100 plus different memory hierarchy. This is the new floor on this hardware.
- Quick probe had a misleading cold-start burst (4 requests during CUDA-graph capture inflated TTFT p50 to 21s); steady-state poisson/constant at ~3.86x already foreshadowed the full-eval improvement.
- FP8 weights did not hurt quality at all (ratio 1.0 at n=500). Confirms FP8 weights are a free knob on Mistral-7B-Instruct-v0.3 for this scenario.

### Suggested follow-ups (queued)

- Multi-step scheduler (`--num-scheduler-steps 8`) — assigned as PR #129.
- Compare `--block-size 32` vs default 16 for cache locality at c=64.
- Test launcher without prefix caching to isolate whether LongBench prompts actually share prefixes after templating.
- Lower `--max-num-seqs` to 192 if scheduler thrash dominates burst phase.

---

## 2026-05-27 17:17 — PR #126: Scenario A — vLLM chunked prefill + FP8 for TTFT

- Branch: `frieren/scenario-a-chunked-prefill-fp8`
- Student: frieren
- Status: **MERGED — first measured Scenario A row in BASELINE.md**

### Hypothesis

Scenario A is input-heavy (128 requests, concurrency=1, 8K input, 128 output). Primary metric is `scenario/A/speedup_over_pytorch` based on inverse TTFT p50. Test whether vLLM with `--enable-chunked-prefill`, `--max-num-batched-tokens=16384`, `--max-num-seqs=16`, `--quantization fp8`, and `--gpu-memory-utilization=0.90` beats the PyTorch baseline on TTFT.

### Results

| Run | Eval mode | Primary metric | Value | Quality ratio | Quality pass | terminal_eligible |
|---|---|---|---:|---:|---|---|
| `u2dxy31b` | quick | scenario/A/speedup_over_pytorch | 1.91x | 0.839 (n=16 screen) | screening-only | False |
| `u44zjwyh` | full | scenario/A/speedup_over_pytorch | **1.893x** | **1.0** (n=500) | True | **True** |

- Speed requests: 128/128 succeeded, 0 failures.
- Validation: `validation_pass=true`, `baseline_update_allowed=true`.
- VRAM peak: 90.79 GiB on RTX PRO 6000 (~96 GiB available).
- TTFT p50: 0.2316 s vs PyTorch 0.4385 s. p90/p99 within 13 ms of p50 — extremely tight tail.

### Analysis

- The 1.893x speedup is the **first measured Scenario A baseline on RTX PRO 6000**.
- FP8 weights had zero quality impact (ratio=1.0 at n=500). Quick-probe quality miss (0.839 ratio at n=16) was pure sampling noise — 1-question gap out of 16.
- Chunked prefill + 16384 batched tokens allows the full 8K prefill in a single scheduler step. TTFT tail is remarkably flat (p99 only 37 ms above p50), suggesting no scheduling stall at any point in the 128-request sequence.
- Reference gap: H100 SMAC3-tuned vLLM = 4.37x vs our 1.89x. The remaining headroom is significant. Bottleneck is now prefill compute throughput, not scheduler or quantization overhead.
- BF16 FP8 decomposition note: FP8 KV cache was not used (RTX PRO 6000 FlashAttention incompatibility); only FP8 weight quant. Heavier quantization (INT4/AWQ) could push further.

### Suggested follow-ups

1. **N-gram speculative decoding on Scenario B** — assigned as PR #131. At c=1 with 8K output, spec-dec directly attacks TPOT.
2. **AWQ/INT4 weight quantization on Scenario A** — halves memory bandwidth vs FP8, potential for another 1.5-2x TTFT improvement.
3. **Speculative decoding on Scenario A** — marginal benefit (TTFT is prefill-bound, spec-dec helps decode), but worth a quick probe.
4. **EAGLE3 or Medusa heads** — requires separate draft head checkpoint but could enable 3-5x spec acceptance rate at c=1.

---

## 2026-05-27 17:27 — PR #127: Scenario B — vLLM n-gram spec decoding + FP8 (research signal)

- Branch: `fern/scenario-b-ngram-spec-decoding`
- Student: fern
- Status: **CLOSED — research signal, terminal_eligible=false, baseline_update_allowed=false (screening_only quality)**

### Hypothesis

Scenario B is output-heavy (64 requests, c=1, 1K input, 8K output). TPOT is the primary metric. At c=1 with long decode sequences, n-gram speculative decoding should dramatically cut TPOT. Two arms: ngram-spec-fp8 (FP8 weights + ngram) and ngram-spec (BF16 weights + ngram after FP8 quality failure).

### Results

| Arm | Launcher | TPOT p50 | speedup | quality ratio (n=16) | W&B | terminal_eligible |
|---|---|---:|---:|---:|---|---|
| PyTorch baseline | — | 25.15 ms | 1.00x | 1.00 | — | baseline |
| ngram-spec-fp8 | `B/ngram-spec-fp8/start_server.sh` | 7.14 ms | **3.52x** | 0.629 ❌ | ext104re | False |
| ngram-spec (BF16) | `B/ngram-spec/start_server.sh` | 7.17 ms | **3.51x** | 0.839 ⚠️ | 5b0w8j17 | False |

- Eval mode: quick (n=4 speed, n=16 quality) — screening only, not eligible for BASELINE update.
- VRAM: ~87-88 GiB (above H100 80 GB envelope — needs gpu-mem-util reduction for leaderboard comparability).

### Analysis

- N-gram speculation cuts TPOT from 25ms → 7ms (3.5x), confirming the hypothesis. Speculation works on RTX PRO 6000 / vLLM 0.11.0 v1 using `--speculative-config '{"method":"ngram",...}'` JSON form.
- FP8 weights on Scenario B: **zero measurable speedup over BF16** (7.14 vs 7.17 ms TPOT, within noise), but worse quality screening (0.629 vs 0.839 ratio). FP8 is not appropriate for Scenario B — use BF16 weights + ngram.
- Quality screening (n=16) is unreliable — fern's binomial analysis is correct: P(≤4/16 | p=0.298) ≈ 0.59, so 4/16 is the *expected* outcome under H₀, meaning BF16+ngram should pass the full n=500 gate.
- VRAM ~88 GB exceeds H100 80 GB budget; `gpu-memory-utilization 0.90` is too generous for this hardware. Reduce to ~0.75 for the next launch.

### Key lessons banked

- N-gram speculative decoding at c=1: 3.5x TPOT improvement confirmed on Mistral-7B-Instruct-v0.3.
- FP8 weight quantization on decode-heavy Scenario B: dominated (0 benefit, quality cost). Drop FP8 from all future Scenario B candidates; use BF16 + ngram.
- `--speculative-config` JSON form works in vLLM 0.11.0 v1 on RTX PRO 6000.
- `num_speculative_tokens=5` is the test configuration; sweep {3, 5, 7} in next launch.
- `gpu-memory-utilization=0.75` target for H100 80GB portability.

### Queued for next launch (high priority)

1. **Full eval of `B/ngram-spec/start_server.sh`** (BF16 + ngram k=5, gpu-mem-util=0.75) — should pass quality and yield first measured Scenario B baseline.
2. **`num_speculative_tokens` sweep** {3, 5, 7} after BF16 base lands.
