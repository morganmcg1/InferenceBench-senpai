# SENPAI Research State

- **Updated:** 2026-05-24 ~17:10
- **Latest human directive:** Infrastructure hotfixes — FlashInfer JIT and FP8 KV
  are **confirmed broken** on RTX PRO 6000 Blackwell in this pod. All launchers
  must avoid `--kv-cache-dtype fp8` and `VLLM_ATTENTION_BACKEND=FLASHINFER`.
  `runtime_env.sh` patched with `VLLM_USE_FLASHINFER_SAMPLER=0`,
  `VLLM_DISABLE_FLASHINFER_PREFILL=1`, `INFERENCE_BENCH_MAX_MODEL_LEN=32768`.
- **Research focus:** LLM inference-serving optimization on InferenceBench.
  Per-scenario speedup over the PyTorch baseline (Mistral-7B-Instruct-v0.3) is
  the paper-facing metric. Quality (MMLU-Pro tau=0.95) is a hard gate.
- **Hardware context:** RTX PRO 6000 Blackwell, 96 GB VRAM, seed 248 (shakedown,
  not leaderboard-comparable). H100 repeat required for leaderboard claims.
- **Pod packing:** 3 students share 1 GPU. Coordination via `senpai/gpu_slot.py`
  with `--wait` flag.

## Merged winners

- **#61 r4-fern — Scenario A:** 1.261x (TTFT p50 0.348 s). vLLM, BF16 weights,
  auto KV, max-seqs 16, batched-tokens 16384, chunked prefill + prefix caching.
- **#59 r4-tanjiro — Scenario C:** 22.23x geomean (burst 32.4x, poisson 23.4x,
  constant 14.5x). vLLM, BF16 weights, auto KV, max-seqs 256, batched-tokens
  8192, block-size 16, chunked prefill + prefix caching.

## Active hypotheses

- **#62 r4-frieren — Scenario B (output-heavy / TPOT).** WIP. vLLM n-gram
  speculative decoding (5 spec tokens, lookup 2-4) with `--kv-cache-dtype auto`.
  Corrected launcher posted (FP8 KV removed). Eval in progress on shared GPU.
- **#69 r4-fern — Scenario A, FP8 weight quantization.** Round 2. Extends
  merged winner (#61, 1.261x) by adding `--quantization fp8` (weight quant,
  not KV). Must beat 1.261x and pass quality. FP8 weights are **likely broken
  on SM120 Blackwell** per researcher note; rescue arm removes quantization.
- **r4-tanjiro — Scenario D (balanced).** Round 2 assignment in flight. No
  candidate currently; PyTorch baseline only. Recipe should combine PR #61's
  prefill levers (chunked prefill, prefix caching, batched-tokens 16384) with
  PR #59's concurrency levers (max-seqs scaled to 32 for 4k/2k workload at
  conc=4) while keeping BF16 + auto KV.

## Aggregate scoreboard (current best, this advisor branch)

- Geomean of merged + PyTorch placeholders = geomean(1.261, 1.0, 22.23, 1.0) ≈
  2.18x. SMAC3 H100 reference is 11.53x aggregate; vLLM-default H100 is 4.05x.
  Headroom is concentrated in **B and D**, plus a likely 2-3x ceiling lift in
  A from kernel-level work (SGLang Triton or INT4/8 weights).

## Next potential directions (round 3+)

Given confirmed constraints (no FP8 KV, no FlashInfer on this pod):

- **Scenario A (prefill, current 1.261x):**
  - If FP8 weights (#69) work on SM120 → quick win. If not → SGLang with
    Triton attention (avoids FlashAttention/FlashInfer entirely, Triton
    radix-tree prefix cache).
  - INT8 or INT4 weight quantization via vLLM's `--quantization gptq`,
    `--quantization awq` (AWQ-Marlin kernel is BF16/INT4, no FP8 codepath).
  - `--no-enable-chunked-prefill` at conc=1 for a single 8k prefill pass —
    chunking overhead may not be amortized at this batch size.
- **Scenario B (decode, current PyTorch):**
  - n-gram speculative (#62) — must survive quality gate.
  - If #62 fails quality → MEDUSA-style draft-and-verify, or `--enable-chunked-prefill`
    for kept-warm decode batches.
  - Weight quantization (INT4/INT8 AWQ) for decode bandwidth gain.
  - `--max-num-seqs 32` to allow overlapped decode batches across burst conc=1
    when the n-gram speculator stalls on a verify step.
- **Scenario C (throughput, current 22.23x):**
  - Push `--max-num-seqs 384/512` with same batched-tokens; verify VRAM headroom
    (Tanjiro's run hit ~91 GB peak with 256).
  - `--max-num-batched-tokens 16384` to feed prefills faster at burst c=64.
  - SGLang radix-tree prefix cache vs vLLM hash prefix at high concurrency.
- **Scenario D (balanced, current PyTorch):**
  - Round 2 assignment to r4-tanjiro — combine A and C learnings.
  - Geomean(1/TTFT.p50, 1/TPOT.p50, req/s) — every dimension counts equally,
    no single lever wins; needs balanced concurrency + prefill scheduling.
- **SGLang exploration:** Triton attention is Blackwell-compatible; avoids the
  broken CUDA-JIT FlashInfer path. Worth trying when a student frees up.

## Hardware constraint summary (RTX PRO 6000 Blackwell, this pod)

- **BROKEN:** `--kv-cache-dtype fp8`, `VLLM_ATTENTION_BACKEND=FLASHINFER`
- **LIKELY BROKEN (untested in this run):** `--quantization fp8` weight quant
  on SM120 Blackwell (per researcher note; PR #69 tests this with a rescue arm)
- **SAFE:** `--kv-cache-dtype auto`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN`,
  `--quantization awq` (BF16/INT4, Marlin kernel), `--quantization gptq`
- **MAX_MODEL_LEN:** 32768 (set by runtime_env.sh)
