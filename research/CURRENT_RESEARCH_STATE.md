# SENPAI Research State
- 2026-05-28 18:20 UTC (round 1 closing in ~17 min)
- No messages from human researcher team this round.

## Round-1 outcome
**Exploration-only round, no terminal winner.** Full Scenario B eval
takes ~57-60 min on RTX PRO 6000 with the leading ngram-k=5 launcher
(64 requests × ~7400 output tokens × 7.2 ms/tok, `ignore_eos=true`);
students continued quick exploration past T+90min and no full eval was
launched. All quick evidence is preserved in `/BASELINE.md` provisional
ledger.

## Strongest screening candidates for next round
1. **F1 — BF16 + ngram k=5 (lookup_max=4, min=2)**: 3.50x speedup, quality
   ratio 0.84 (n=16, matches vLLM default screening floor → consistent
   with noise, not degradation). Lowest-risk terminal candidate.
2. **F4 — BF16 + ngram spec=25 lookup=12**: 5.60x speedup but quality
   ratio 0.63 (n=16). Higher peak speed; quality at n=500 unknown.
   Higher-risk/higher-reward.
3. **F2 — BF16 + ngram k=3**: 3.28x, quality 0.63. Mid-range, probably
   dominated by F1.

## Key negative findings (do not re-test in next round without new evidence)
- **FP8 weight quant (`--quantization fp8`) HURTS at decode on RTX PRO
  6000 / vLLM 0.11**. Two independent measurements (F3 vs F1; N3 vs F2)
  both show FP8 slower. Mechanism: dequant overhead dominates the
  bandwidth saving at single-stream decode.
- **vLLM FlashInfer must stay disabled** on this hardware (per
  `runtime_env.sh`); not retested.

## Next-round priorities

### Process changes (MUST)
1. **Hard advisor gate at T+30 min**: after initial quick probes,
   advisor picks the candidate; student MUST start full eval immediately.
2. **Force launcher git push** before any eval slot acquisition.
3. **Use medium-screening eval (n=64-100 quality)** for arm
   discrimination instead of n=16 quick, to reduce false-positive
   quality scares (F2/F4 n=16 quality of 0.63 may have been noise).

### Promising hypotheses (rank-ordered)
1. **F1 BF16+ngram k=5 → full eval**. Highest priority — known 3.50x,
   quality probably clears n=500 since it matches default floor.
2. **F1 + lean scheduler (≈ fern's N4 launcher already committed at
   `senpai/launchers/scenario_b/n4_bf16_ngram5_minsched/`)**: tests
   whether scheduler-minimization stacks orthogonally onto ngram-k=5.
3. **Compile path**: `--compilation-config` or torch.compile piecewise
   CUDA graphs on the F1 launcher.
4. **TRITON_ATTN backend** vs default attention on this Blackwell GPU
   (one-flag experiment).
5. **medium-spec window survey** (k=7, k=10, k=15) with
   `prompt_lookup_max` matched conservatively, on n=64 quality
   screening, to find the speed/quality knee.

### Lower priority / blocked
- SGLang / TGI engine swap: needs venv install time we don't have inside
  a 2h round; would need a separate engine-prep PR.
- FP8 KV cache: blocked by FLASH_ATTN incompatibility; would need
  FLASHINFER and a careful boot test.
- Custom kernels: out of scope for a 2h SENPAI round.

## Current baseline
No terminal best on this branch. Provisional best quick: F1 at 3.50x
speedup, screening quality ratio 0.84 (n=16). See `/BASELINE.md`.
