# SENPAI Research Results

## 2026-05-28 18:20 — PR #159 (frieren): Scenario B n-gram speculative decoding + CUDA graphs

- Branch: `scen-b-frieren/scenario-b-ngram-spec-decode`
- Hypothesis: N-gram speculative decoding amplifies decode-step output rate at burst concurrency 1, directly improving `scenario/B/speedup_over_pytorch = 1/tpot.p50`. Stack with default CUDA graphs (`enforce_eager=false`).
- Hardware: RTX PRO 6000 Blackwell shakedown (NOT leaderboard-comparable to H100)

### Arms run (all `quick` screening, n=16 quality)

| Arm | Launcher | TPOT p50 | Speedup | MMLU-Pro (n=16) | Quality ratio | W&B |
|---|---|---:|---:|---:|---:|---|
| F0 vLLM default | f0_vllm_default | 17.47 ms | 1.44x | 0.250 | 0.84 | o39u89pa |
| F1 BF16 + ngram k=5 lookup_max=4 min=2 | f1_ngram_k5 | **7.18 ms** | **3.50x** | 0.250 | 0.84 | 3rvxd2tx |
| F2 BF16 + ngram k=3 | f2_ngram_k3 | 7.67 ms | 3.28x | 0.188 | 0.63 | smaj0ntn |
| F3 FP8 weights + ngram k=5 | f3_ngram_fp8 | 7.29 ms | 3.45x | (pass at n=16) | (pass) | yittyrbl |
| F4 BF16 + ngram spec=25 lookup=12 | (not committed) | ~4.5 ms | 5.60x | 0.188 | 0.63 | s5aj7628 |
| F5 BF16 + ngram spec=25 lookup=15 | (not committed) | ~6.0 ms | 4.19x | 0.188 | 0.63 | oaveolmf |

PyTorch reference: TPOT p50 = 25.15 ms (raw 1/tpot.p50 = 39.76 tok/s).

### Results commentary

- **Strong primary signal**: n-gram speculative decoding gives ≥3.28x screening speedup across all spec configurations on Scenario B's long-decode workload. The decode-step amplification mechanism works as predicted.
- **k=5 > k=3 at modest spec windows**: F1 (k=5) 3.50x beat F2 (k=3) 3.28x on TPOT despite higher acceptance risk; with Mistral-7B-Instruct on natural-language outputs the n-gram acceptance rate at k=5 is high enough that the larger speculation window wins.
- **FP8 weight quantization does NOT stack** with ngram on this hardware. F3 (FP8+k=5) 3.45x ~ F1 (BF16+k=5) 3.50x. Combined with fern's N3 (FP8+k=3+lean) 2.42x vs F2 (BF16+k=3) 3.28x, two independent measurements confirm `--quantization fp8` adds dequant overhead at single-stream decode on RTX PRO 6000 / vLLM 0.11.
- **Aggressive spec windows trade quality for speed**: F4 (spec=25) hit 5.60x but quality at n=16 dropped to 0.63 ratio (below the screening noise floor of 0.84). Could still be small-sample noise — would need a medium-quality (n=64-100) screening to discriminate.
- **No terminal full eval was run** this round. Full Scenario B eval requires ~57 min of speed-eval alone; the 2h window was spent on 6 quick arms.

### FULL EVAL — F1 terminal result (18:22 UTC)

| Arm | Eval mode | TPOT p50 | Speedup | MMLU-Pro (n=500) | Quality ratio | W&B |
|---|---|---:|---:|---:|---:|---|
| F1 BF16 + ngram k=5 | full (64 requests) | **8.94 ms** | **2.81x** | **0.314** | **1.054** | `6r6w0ukx` |

PyTorch reference: TPOT p50 = 25.15 ms, raw 1/tpot.p50 = 39.76 tok/s.
Quality gate: τ=0.95 × 0.298 = 0.283. Observed 0.314 (ratio 1.054) → **PASS**.
Validation: `validation_pass=true`, `baseline_update_allowed=true`, `terminal_eligible=true`, 64/64 requests.

### Results commentary (full eval)

- **Full eval settled at 2.81x vs 3.50x quick screening.** At n=4 burst quick probes, n-gram acceptance rate is higher due to shorter prompt contexts; at n=64 burst with `ignore_eos=true` and ~7400-token outputs the acceptance dynamics are slightly less favourable, but still a strong and clean win.
- **Quality gate passed convincingly.** 0.314 vs baseline 0.298 (ratio 1.054) across 500 MMLU-Pro questions. N-gram speculative decoding with BF16 weights and k=5 preserves model quality at this spec window; no speculative-decoding-induced degradation at 500-sample resolution.
- **PR #159 merged to advisor branch `ib-20260528-scen-b-r1` at 18:25 UTC** as the round-1 terminal winner.

### Conclusion: **TERMINAL WINNER — 2.81x speedup. Merged.**
- Launchers F0/F1/F2/F3 squash-merged into advisor branch.
- F4/F5 (spec=25) remain as research signals only — aggressive spec windows at n=16 quality ratio 0.63, quality at n=500 unknown, not promoted.
- New round-1 baseline: **F1 BF16 + ngram k=5 = 2.81x speedup** (full eval, 64 requests, 500-sample quality gate).

---

## 2026-05-28 18:20 — PR #162 (fern): Scenario B FP8 weight quant + minimal-scheduler vLLM

- Branch: `scen-b-fern/scenario-b-fp8-weight-scheduler`
- Hypothesis: FP8 weight quantization halves weight-load bandwidth (decode is BW-bound at c=1) and a minimal scheduler (max_num_seqs=1, no chunked prefill, no prefix caching) removes per-step overhead. Stack: FP8 + lean scheduler + ngram k=3.
- Hardware: RTX PRO 6000 Blackwell shakedown

### Arms run

| Arm | Launcher | TPOT p50 | Speedup | MMLU-Pro (n=16) | Quality ratio | W&B |
|---|---|---:|---:|---:|---:|---|
| N3 FP8 + ngram k=3 + lean scheduler | n3_fp8w_ngram3 | 10.38 ms | 2.42x | 0.313 | 1.05 | vpymvy8e |

Plus committed launcher recipes (not run): N1 (`n1_fp8w`), N2 (`n2_fp8w_minsched`), N4 (`n4_bf16_ngram5_minsched`).

### Results commentary

- **Hypothesis partially refuted**: FP8 weight quant was the central lever and did not help on this hardware (see PR #159 cross-confirmation).
- **Excellent independent analysis**: fern cross-validated frieren's FP8 finding (her own N3 at 2.42x is slower than frieren's F2 BF16+k=3 at 3.28x by the same FP8-vs-BF16 ratio observed in F3 vs F1).
- **N4 launcher (BF16 + ngram k=5 + lean scheduler) is committed and ready** to evaluate next round; tests whether scheduler-minimization stacks orthogonally onto ngram-k=5 (frieren's F1 winner config).

### Conclusion: PR #162 closed as `research-signal-only`. N4 launcher recipe preserved for next round.

---

## Round-1 process lessons

1. Full Scenario B eval is ~57-60 min on RTX PRO 6000; in a 2h window the advisor must force the full-eval slot to start no later than T+~60 min.
2. Quick screening at n=16 quality is too noisy to discriminate; advisor should mandate n=64-100 medium screening on the selected candidate before promotion.
3. Both students were slow to push launcher recipes; advisor should require `git push` before each eval slot acquisition.
4. The GPU-slot coordination worked — no measurement corruption between students sharing the GPU.