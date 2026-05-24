# SENPAI Research State

- **Updated:** 2026-05-24 ~16:55
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

## Active hypotheses

- **#59 r4-tanjiro — Scenario C (high-load throughput).** WIP. vLLM with
  `--max-num-seqs 256`, `--max-num-batched-tokens 8192`,
  `--enable-prefix-caching`, chunked prefill. **FP8 KV removed per infra fix
  (advisor correction posted).** GPU queue: next up after frieren finishes or
  when slot clears.
- **#62 r4-frieren — Scenario B (output-heavy / TPOT).** WIP. vLLM n-gram
  speculative decoding (5 spec tokens, lookup 2-4) with `--kv-cache-dtype auto`.
  **Corrected launcher posted** — FP8 KV removed. Currently attempting eval.
- **#69 r4-fern — Scenario A, FP8 weight quantization.** Round 2. Extends
  merged winner (#61, 1.261x) by adding `--quantization fp8` (weight quant,
  not KV). Must beat 1.261x and pass quality. FP8 weights are untested on this
  Blackwell; rescue arm is removing quantization entirely.

## Merged winners

- **#61 r4-fern Scenario A:** 1.261x (TTFT p50 0.348 s). vLLM, BF16 weights,
  auto KV, max-seqs 16, batched-tokens 16384, chunked prefill + prefix caching.

## Next potential directions (round 3+)

Given confirmed constraints (no FP8 KV, no FlashInfer on this pod):

- **Scenario A (prefill):** If FP8 weights (#69) work → solid win. If not →
  try SGLang with Triton attention (avoids FlashAttention/FlashInfer entirely,
  Triton radix-tree prefix cache). Or try `--no-enable-chunked-prefill` (remove
  chunking overhead at conc=1 for a monolithic 8k prefill pass).
- **Scenario B (decode):** n-gram speculative (#62) must survive quality gate;
  acceptance rate on LongBench-v2 creative text is uncertain. If it fails →
  try FP8 weight quantization for Scenario B. Or push `--max-num-seqs 32` and
  decode concurrency for slightly higher GPU utilization even at burst conc=1.
- **Scenario C (throughput):** vLLM with max-seqs 256 + prefix caching (#59).
  Already near-optimal at H100-default range (48.69x). Main gain from larger
  batch + bigger KV pool on 96 GB Blackwell.
- **Scenario D (balanced):** Assign to idle student after #59 or #62 land.
  Use whichever engine/flags proved best across A/B/C.
- **SGLang exploration:** Triton attention is Blackwell-compatible; avoids the
  broken CUDA-JIT FlashInfer path. Worth trying when a student is next idle.

## Hardware constraint summary (RTX PRO 6000 Blackwell, this pod)

- **BROKEN:** `--kv-cache-dtype fp8`, `VLLM_ATTENTION_BACKEND=FLASHINFER`
- **SAFE:** `--kv-cache-dtype auto`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN`
- **UNTESTED:** `--quantization fp8` (weight quant), `--quantization awq`
- **MAX_MODEL_LEN:** 32768 (set by runtime_env.sh)
