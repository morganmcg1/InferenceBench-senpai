# SENPAI Research State

- **Updated:** 2026-05-28T17:40Z
- **Most recent direction from human researcher team:** No GitHub Issues. Run
  scope fixed by operator: Scenario A parity launch, RTX PRO 6000 shakedown,
  2h budget, 1 GPU shared, 2 logical students.

## Terminal winner (merged)

**PR #160 — scen-a-fern/fern-fp8-weights — 1.8873x speedup — MERGED to `ib-20260528-scen-a-r1`**

vLLM `--quantization fp8` on-the-fly weight quantization. Full eval 128/128
success, MMLU-Pro ratio 0.9597 (pass, tight but valid), TTFT.p50 0.2325s, W&B
`bt2ckkzz`. BASELINE.md updated.

## Active probes (quick-only, no full eval window)

| PR | Student | Arm | What it tests | Expected | W&B group |
|---|---|---|---|---|---|
| #168 | scen-a-fern | C0 FlashInfer prefill + FP8 (sampler disabled) | Does FlashInfer's paged-attention kernel beat FlashAttention-2 at 8192-token prefill? | 10-30% potential TTFT improvement if kernel is faster | `fern-flashinfer-fp8` |
| #168 | scen-a-fern | C1 FlashInfer + FP8 + FP8 KV (only if C0 boots) | Does FP8 KV cache help with FlashInfer backend? | Small additional gain | `fern-flashinfer-fp8` |
| #169 | scen-a-frieren | D0 FP8 + enforce_eager (no CUDA graphs) | Does CUDA graph dispatch overhead matter at concurrency 1 with FP8? | Neutral to small win | `frieren-eager-fp8` |

PR #158 (BF16 prefill tuning) was closed after establishing the BF16 ceiling
at ~1.276x. The chunked-prefill toggle and max-model-len tightening are
confirmed non-levers at burst concurrency 1.

## Key findings so far

- **FP8 weights are the dominant Scenario A lever** on RTX PRO 6000. Going
  from BF16 (1.276x) to FP8 weight GEMM (1.887x) is a ~48% relative TTFT
  reduction.
- **Prefill structure tuning (chunked-prefill, max-model-len, batched-tokens)
  does not move TTFT** at concurrency 1 with 8192-token inputs — in both BF16
  (PR #158) and FP8 (B0 vs B1, PR #160). The bottleneck is the GEMM, not
  scheduling.
- **FlashInfer sampler JIT fails on this image** (missing `curand.h`).
  Workaround: set `VLLM_USE_FLASHINFER_SAMPLER=0` and
  `VLLM_DISABLE_FLASHINFER_SAMPLING=1` explicitly inside the launcher.
  FlashInfer **prefill** does not have this dependency.

## Time budget (as of 17:40Z)

- Cutoff: `2026-05-28T18:36:57Z` → **~56 min remaining**.
- Reserve 10-15 min for final review/merge/BASELINE: effective student window
  ends ~18:22-18:27.
- Quick probes (PR #168 C0, C1): ~30 min total → finishes ~18:10.
- PR #169 D0 (frieren, queued after #168): ~15 min → finishes ~18:25.
- A full eval on PR #168 is only feasible if C0 shows ≥15% quick improvement
  over current best (TTFT ≤ 0.197s) AND there are ≥50 min remaining at
  the time the quick result lands (that window has essentially closed unless
  C0 is very fast).

## Next-round directions (if another launch runs)

1. **FlashInfer prefill + FP8 weights**: if today's C0 shows headroom, this
   is the first thing to full-eval next launch.
2. **vLLM V0 engine + FP8**: VLLM_USE_V1=0 with fp8 weights as engine-
   version comparison.
3. **SGLang Triton backend + FP8**: engine-family diversity for long-context
   prefill.
4. **FP8 + `torch.compile` optimization level**: vLLM 0.11 supports compile
   modes that may further fuse prefill ops.
5. **MMLU-Pro quality margin improvement**: the current winner is at 0.9597
   quality ratio (just above 0.95 floor). A quantization scheme with better
   quality preservation (e.g. GPTQ instead of naive FP8) may both pass quality
   more safely and maintain speed.
6. **H100 leaderboard run**: repeat winning FP8 recipe on H100 for
   leaderboard-comparable claim.

## Infrastructure lessons from this run

- Explicit `VLLM_USE_FLASHINFER_SAMPLER=0` / `VLLM_DISABLE_FLASHINFER_PREFILL=1`
  / `VLLM_DISABLE_FLASHINFER_SAMPLING=1` must be in each launcher (env does not
  persist through gpu_slot shell).
- FP8 KV cache requires FlashInfer backend (FlashAttention rejects it).
- `senpai/require_scoring_preflight.sh --import-dir ...` hydration works
  cleanly for RTX PRO 6000 seed 248 assets.
