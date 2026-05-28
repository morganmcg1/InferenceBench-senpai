# SENPAI Research State
- 2026-05-28 18:26 UTC
- No messages from human researcher team this round.

## Round-1 outcome
**Terminal winner confirmed: F1 (BF16 + ngram k=5) at 2.81x speedup.**
Frieren ran a full eval at T+~100 min (just before the 18:36:57 UTC deadline).
PR #159 merged at 18:25 UTC. New baseline: `scenario/B/speedup_over_pytorch = 2.81`.

Previous PyTorch reference: 1.00x (TPOT p50 = 25.15 ms, 1/tpot.p50 = 39.76 tok/s).

## Current best result
- **F1 BF16 + ngram k=5** (lookup_max=4, min=2): 2.81x speedup, TPOT p50 = 8.94 ms
- Quality: MMLU-Pro n=500 accuracy 0.314 (ratio 1.054, τ=0.95 → pass)
- Launcher: `senpai/launchers/scenario_b/f1_ngram_k5/start_server.sh`
- W&B: `6r6w0ukx`
- Note: quick screening reported 3.50x (n=4); full 64-request burst settled at 2.81x.

## Headroom analysis
- H100 public reference (SMAC3 2h search): 15.23x — **5.4× higher than our current 2.81x**
- H100 vLLM default: 2.25x. Our RTX PRO 6000 F1 = 2.81x already beats H100 vLLM default.
- Key gap: spec-decode acceptance rate and hardware throughput differ substantially between
  RTX PRO 6000 Blackwell and H100. On-hardware full-eval tuning is the path forward.

## Key negative findings (do not re-test without new evidence)
- **FP8 weight quant HURTS** at single-stream decode on RTX PRO 6000 / vLLM 0.11.
  Two independent measurements (F3 vs F1; N3 vs F2) both show FP8 slower.
  Mechanism: dequant overhead dominates bandwidth saving at single-stream decode.
- **vLLM FlashInfer must stay disabled** on this hardware (per `runtime_env.sh`).

## Next-round priorities (rank-ordered)

1. **N4: F1 + lean scheduler** (`senpai/launchers/scenario_b/n4_bf16_ngram5_minsched/`
   already committed on fern's branch). Tests whether scheduler-minimization
   (max_num_seqs=1, no chunked prefill, no prefix caching) stacks orthogonally with
   ngram-k=5. If it does, target should be ≥3x at full eval.
2. **Spec-window sweep (k=7, k=10, k=15)** on n=64 quality screening to find the
   speed/quality knee. F4 (spec=25) showed 5.60x quick but quality ratio 0.63 at
   n=16; intermediate windows may clear the quality gate with more headroom.
3. **Compile path**: `--compilation-config` or torch.compile piecewise CUDA graphs
   stacked on F1.
4. **TRITON_ATTN backend** vs default flash-attn2 on this Blackwell GPU (one flag).
5. **Engine swap (SGLang)**: needs a separate env-prep PR before experimentation.

## Process improvements for next round
1. **Hard advisor gate at T+30 min**: pick winner from quick probes, start full eval
   immediately. Round-1 frieren almost missed the window by continuing quick exploration.
2. **Force launcher push before each eval slot** — frieren's launchers were pod-local
   until T+~100 min.
3. **Medium screening (n=64-100)** before full eval promotion to reduce quality noise
   (n=16 quality ratios had high variance: 0.63 vs 0.84 at comparable configs).
