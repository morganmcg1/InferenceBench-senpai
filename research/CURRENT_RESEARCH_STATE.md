# SENPAI Research State — ib-20260524-hardened-r5

- **Date**: 2026-05-24 (RTX PRO 6000 shakedown launch, 2 hour budget)
- **Most recent human directive**: None at start; check GitHub Issues each loop.
- **Active scoring assets**: `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248` (preflight pass, exit_code=0)

## Critical hardware constraints (RTX PRO 6000, SM120f, CUDA 13.2, vLLM 0.11)

Discovered live during this run. Apply to ALL subsequent assignments:
- **FlashInfer blocked**: `VLLM_ATTENTION_BACKEND=FLASHINFER` → JIT fails due to CPATH CUDA version mismatch. `runtime_env.sh` now sets `VLLM_USE_FLASHINFER_SAMPLER=0` and `VLLM_DISABLE_FLASHINFER_PREFILL=1`.
- **FP8 KV blocked**: `--kv-cache-dtype fp8` raises `NotImplementedError` on FLASH_ATTN for SM120f. Use `auto`.
- **FP8 weights OK**: `--quantization fp8` boots and delivers real speedup (confirmed by PR #63).
- **Max model len**: Use 32768 or 16384, not 131072.
- **Workaround**: `unset CPATH` before FlashInfer to resolve header conflicts (PR #63 launcher documents this for future use when toolchain is fixed).

## Research focus and themes

The portfolio attacks B (7× gap), D (3× gap), A (4× gap) over PyTorch.
Runtime constraints have redirected lever selection: FP8 weights + CUDA graphs
+ n-gram speculative decoding are the proven-bootable combination.

- Scenario A: **merged** (PR #63, r5-fern) — 1.889× with FP8+FLASH_ATTN. The FlashInfer+FP8-KV gap means there is still 2–2.5× more headroom to recover.
- Scenario B: active (PR #73 r5-fern, PR #60 r5-frieren stalled). n-gram spec + FP8 weights is the highest-leverage available stack.
- Scenario D: active (PR #65 r5-tanjiro). n-gram spec + chunked prefill (no FP8 quant dropped as a safety measure by student).

## Open hypotheses

| Student | Scenario | PR | Status | Lever stack |
|---|---|---|---|---|
| r5-fern    | B | #73 | wip | FP8 weights + n-gram spec + FLASH_ATTN + auto KV |
| r5-tanjiro | D | #65 | wip | n-gram spec + chunked prefill + FLASH_ATTN + auto KV |
| r5-frieren | B | #60 | wip (stalled, 0 commits) | Needs urgent intervention |

## Potential next research directions

Prioritized by expected impact given confirmed hardware constraints:

1. **Scenario A: Triton attention + FP8 KV** — student (fern) suggested vLLM has a Triton attention path that accepts FP8 KV on Blackwell. Would recover the main blocked lever for A.
2. **Scenario B: n-gram spec acceptance sweep** — vary `num_speculative_tokens` (3, 5, 7) to find peak acceptance rate on output-heavy decode.
3. **AWQ INT4 on Scenario B** — often beats FP8 on decode-heavy workloads; worth a head-to-head once FP8 results are in.
4. **Scenario D: FP8 weights reintroduction** — tanjiro dropped FP8 quant as safety measure; if n-gram spec results are in, add FP8 weights on top.
5. **Fix `senpai/runtime_env.sh` CPATH** — unset CPATH or append (not prepend) the pip CUDA headers to stop poisoning FlashInfer JIT for all students.
6. **AOT FlashInfer wheels for SM120f** — eliminates the entire JIT fragility class.
7. **SGLang head-to-head** on the winning scenario once FP8+spec baseline is established.
8. **Scenario C sweep** — low-priority given H100 reference is already near-saturated, but worth measuring on RTX PRO 6000.
