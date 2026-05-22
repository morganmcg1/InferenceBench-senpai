# InferenceBench Research Ideas — 2026-05-22

## Context

Target: Mistral-7B-Instruct-v0.3, single H100 80GB.
Primary metric: speedup = candidate_raw_objective / pytorch_baseline_raw_objective.
Quality gate: MMLU-Pro accuracy >= 0.95x PyTorch FP16 baseline (do NOT skip this gate).
W&B project: wandb-applied-ai-team/inferencebench-senpai

Reference ceilings to beat (from SMAC3 HPO search):
- Scenario A (TTFT, input-heavy): 4.37x
- Scenario B (TPOT, output-heavy): 15.23x
- Scenario C (req/s, high load): 46.70x — NOTE: SGLang default already 51.12x, vLLM default 48.69x
- Scenario D (geomean, general): 5.69x

---

## Hypothesis 1 — vLLM N-gram Speculative Decoding for Scenario B

**Target scenario:** B (output-heavy: input_len=1024, output_len=8192, concurrency=1, burst)
**Target metric:** TPOT speedup > 15.23x (SMAC3 ceiling)
**Engine:** vLLM

### Why this should work

Scenario B is pure decode-bottleneck: concurrency=1, 8192 output tokens per request, only 1024 input tokens. N-gram speculative decoding (no separate draft model) proposes draft tokens by looking up n-gram matches in the prompt and recent generation. LongBench v2 prompts contain long, repetitive context windows — document QA, summarization, code reading — so the acceptance rate should be high. With concurrency=1 there is no scheduler contention and the speculative overhead is minimal. vLLM's implementation uses a single forward pass for verification, so wall-clock cost per accepted token is well below 1 standard step.

Key literature: vLLM speculative decoding docs; "Accelerating Large Language Model Decoding with Speculative Sampling" (Chen et al., 2023); "Medusa" and "EAGLE" show 2-3x TPOT gains even for non-repetitive text; LongBench-style prompts should push acceptance rate higher.

### Exact configuration

```bash
export VLLM_ATTENTION_BACKEND=FLASHINFER   # faster single-sequence decode on H100
export CUDA_VISIBLE_DEVICES=0

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --block-size 32 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 4096 \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_num_tokens":4}' \
    --trust-remote-code \
    --disable-log-stats
```

Notes:
- `max-model-len 32768` not 131072 — Scenario B only needs 1024+8192; reducing sequence length frees KV cache blocks for more speculative candidates.
- `VLLM_ATTENTION_BACKEND=FLASHINFER` is required for ngram speculation in recent vLLM versions (Flash Attn backend can error on spec-decode with single batch).
- `num_speculative_tokens=5` is the sweet spot from vLLM ablations; 7 regresses when acceptance drops below ~70%.
- `prompt_lookup_num_tokens` (not `prompt_lookup_max`) is the correct key name in vLLM >= 0.4.3 — check the exact installed version and adjust if needed.

### Pitfalls

- Speculative decoding is disabled if `temperature=0` — Scenario B uses temperature=0.7 so this is safe.
- If `ignore_eos=true` is set in the benchmark, EOS token is never emitted, which means speculative decode runs for the full 8192 tokens with no short-circuit. This is expected behavior.
- Quality gate: FP16 model + speculative decode produces identical logit distributions for accepted tokens — quality gate should pass cleanly. Verify MMLU-Pro score before reporting.

### Fast fallback

If server fails to start (speculative-config parse error on older vLLM): drop `--speculative-config` entirely, add `VLLM_ATTENTION_BACKEND=FLASHINFER` only, and re-run. That alone should yield ~1.5-2x TPOT over the plain default.

---

## Hypothesis 2 — vLLM FLASHINFER + FP8 for Scenario A

**Target scenario:** A (input-heavy: input_len=8192, output_len=1024, concurrency=1, burst)
**Target metric:** TTFT speedup > 4.37x (SMAC3 ceiling)
**Engine:** vLLM

### Why this should work

Scenario A is pure prefill-bottleneck: concurrency=1, 8192-token inputs, only 1024 outputs. TTFT is dominated by the time to process the full prompt through all attention layers. Two levers:

1. **FLASHINFER attention backend** — FlashInfer uses tiled GEMM-based attention that is faster than FlashAttention-2 for long-context single-sequence prefill on H100 (confirmed in FlashInfer benchmarks and vLLM docs for H100). 

2. **FP8 weights + FP8 KV cache** — halves the memory bandwidth required for weight reads and KV cache reads. On H100, FP8 matmul runs at ~2x the FLOP rate of BF16. The 7B model fits comfortably in FP8 with no accuracy loss beyond MMLU-Pro gate.

Combination: with FP8 weights and FLASHINFER, each attention layer in the prefill pass runs faster; with FP8 KV cache, the KV write after each layer is cheaper. For a 128-request burst at concurrency=1, this compounds across all requests.

Disable chunked prefill: for single-sequence prefill, chunked prefill adds scheduling overhead with no benefit. Keep it off (default).

### Exact configuration

```bash
export VLLM_ATTENTION_BACKEND=FLASHINFER
export CUDA_VISIBLE_DEVICES=0

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --block-size 16 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 16384 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --disable-log-stats
```

Notes:
- `max-num-batched-tokens 16384` covers the full 8192-token prompt in a single prefill chunk (no chunking needed).
- `block-size 16` is better for FP8 KV cache due to finer-grained memory alignment.
- `quantization fp8` uses vLLM's built-in FP8 dynamic quantization (no separate quantized checkpoint needed — it quantizes at load time).
- `max-num-seqs 4` — keep low since concurrency=1; too many allocated sequence slots wastes KV cache.

### Pitfalls

- FP8 dynamic quantization (`--quantization fp8`) is different from loading a pre-quantized FP8 checkpoint. Quality gate should pass but run MMLU-Pro to confirm.
- FLASHINFER requires `pip install flashinfer` — check whether the starting venv has it. If not, fallback: remove `VLLM_ATTENTION_BACKEND=FLASHINFER` and keep FP8 only.
- `--kv-cache-dtype fp8` with FP8 weights can occasionally trigger numerical issues on some attention head sizes — watch for NaN in first few requests.

### Fast fallback

If FP8 causes quality gate failure: drop `--quantization fp8` and `--kv-cache-dtype fp8`, keep `VLLM_ATTENTION_BACKEND=FLASHINFER` only. FLASHINFER alone should yield meaningful TTFT improvement.

---

## Hypothesis 3 — SGLang LPM Scheduling + High mem_fraction_static for Scenario C

**Target scenario:** C (high load: input_len=1024, output_len=1024, concurrency up to 64 burst / 32 poisson / 16 constant)
**Target metric:** req/s throughput > 51.12x (beat SGLang default, extend past SMAC3 46.70x)
**Engine:** SGLang

### Why this should work

SGLang default is already 51.12x — this is the highest-floor scenario. The SGLang search space allows `schedule_policy=lpm` (Longest Prefix Match) which maximizes KV cache reuse across concurrent requests. LongBench v2 prompts in Scenario C (1024-token inputs) will have shared document prefixes across requests in the same batch, making LPM strictly better than FCFS for cache hit rate and throughput.

High `mem_fraction_static=0.90` gives the KV cache allocator more memory, increasing batch size before eviction occurs. Combined with `max_running_requests=256` to saturate the H100 during burst, this should push req/s higher than the default config.

### Exact configuration

```bash
export CUDA_VISIBLE_DEVICES=0

exec python3 -m sglang.launch_server \
    --model-path "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len 16384 \
    --mem-fraction-static 0.90 \
    --schedule-policy lpm \
    --max-running-requests 256 \
    --chunked-prefill-size 4096 \
    --max-prefill-tokens 32768 \
    --trust-remote-code \
    --disable-flashinfer-sampling
```

Notes:
- SGLang uses `--model-path` not `--model`.
- `--disable-flashinfer-sampling` avoids a known SGLang sampling bug on some H100 firmware versions.
- If SGLang's launcher entrypoint differs (`python3 -m sglang.launch_server` vs `python3 -m sglang.srt.entrypoints.openai.api_server`), check installed version.
- The search space does NOT include quantization for SGLang (only "none") — do not add it.

### Pitfalls

- LPM scheduling requires requests to share prefix structure. If LongBench v2 prompts in Scenario C are fully randomized with no shared prefixes, LPM degrades to FCFS. This is the main risk. Mitigation: even with no cache hits, LPM scheduling is never worse than 5% below FCFS on throughput.
- `mem_fraction_static=0.90` leaves only 10% for activation buffers — may OOM on very large batch sizes. Drop to 0.88 if OOM.

### Fast fallback

If SGLang fails to start (wrong entrypoint, missing package): fall back to vLLM with `--enable-prefix-caching --max-num-seqs 256 --max-num-batched-tokens 16384`.

---

## Hypothesis 4 — TGI FP8 + CUDA Graphs for Scenario D

**Target scenario:** D (general: input_len=4096, output_len=2048, concurrency=4, burst)
**Target metric:** geomean speedup > 5.69x (SMAC3 ceiling)
**Engine:** TGI

### Why this should work

Scenario D is balanced: 4096 input + 2048 output, concurrency=4. TGI with FP8 quantization and CUDA graphs is a configuration orthogonal to what SMAC3 explored (SMAC3 used vLLM and SGLang). TGI's CUDA graph support fuses decode kernel launches, reducing overhead per decode step. FP8 (`--quantize fp8`) on TGI uses the `eetq` or native fp8 path, reducing weight memory bandwidth. With 4 concurrent sequences, CUDA graphs with batch sizes [1,2,4,8,16,32] cover the realistic concurrency range.

TGI's `waiting_served_ratio` controls how aggressively it batches new prefill requests with ongoing decode — setting it to 0.3 minimizes wait and maximizes throughput at concurrency=4.

### Exact configuration

```bash
export CUDA_VISIBLE_DEVICES=0

exec text-generation-launcher \
    --model-id "${MODEL_ID}" \
    --hostname "${HOST}" \
    --port "${PORT}" \
    --max-input-tokens 8192 \
    --max-total-tokens 16384 \
    --max-batch-prefill-tokens 16384 \
    --max-batch-total-tokens 65536 \
    --max-concurrent-requests 64 \
    --max-batch-size 32 \
    --cuda-graphs 1,2,4,8,16,32 \
    --quantize fp8 \
    --cuda-memory-fraction 0.92 \
    --waiting-served-ratio 0.3 \
    --trust-remote-code
```

Notes:
- TGI uses `--hostname` not `--host`.
- `--quantize fp8` requires CUDA >= 11.8 and TGI >= 2.0 — confirm installed version.
- `--max-total-tokens` = input + output + margin: 4096 + 2048 + buffer = 16384 is safe.
- `--cuda-graphs 1,2,4,8,16,32` captures batch sizes for the decode phase; prefill uses a different code path.
- Quality gate: TGI FP8 (`eetq`-based) has very low accuracy loss — MMLU-Pro gate expected to pass.

### Pitfalls

- TGI entrypoint may be `text-generation-launcher` or a Python module depending on install method. Check with `which text-generation-launcher`.
- CUDA graphs compilation on first startup adds ~60-90s cold-start time — do NOT interpret slow startup as a hang.
- FP8 quantize on TGI may not be available if TGI was installed without CUDA FP8 support. Fallback: `--quantize eetq` instead.

### Fast fallback

If TGI is not installed or fails: fall back to vLLM Scenario D with `VLLM_ATTENTION_BACKEND=FLASHINFER --quantization fp8 --max-num-seqs 32 --max-num-batched-tokens 8192 --enable-prefix-caching`.

---

## Hypothesis 5 — vLLM Chunked Prefill Interleaving for Scenario D (Ablation)

**Target scenario:** D (general: input_len=4096, output_len=2048, concurrency=4, burst)
**Target metric:** geomean speedup > 5.69x
**Engine:** vLLM

### Why this should work

At concurrency=4 with 4096-token inputs, chunked prefill allows vLLM to interleave prefill chunks with decode steps from other in-flight requests. This reduces head-of-line blocking: new requests get their first tokens faster, existing decode requests are not stalled waiting for a 4096-token prefill to complete. The combined effect improves both TTFT and TPOT for the mixed workload, boosting geomean speedup.

This is the vLLM complement to Hypothesis 4 (TGI) — run these two back-to-back to compare engines on Scenario D.

### Exact configuration

```bash
export VLLM_ATTENTION_BACKEND=FLASHINFER
export CUDA_VISIBLE_DEVICES=0

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --block-size 32 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --trust-remote-code \
    --disable-log-stats
```

Notes:
- `max-num-batched-tokens 8192` with chunked prefill means each prefill chunk is at most 8192 tokens — covers Scenario D's 4096-token inputs in two chunks max.
- `enable-prefix-caching` helps when multiple Scenario D requests share document prefix (LongBench v2 characteristic).
- No quantization here — keep FP16 to isolate the scheduling effect.

### Pitfalls

- Chunked prefill + FLASHINFER combination had a bug in vLLM < 0.5.0 — verify installed version or test quickly with `--enforce-eager` first.
- At concurrency=4 (burst), chunked prefill may hurt if all 4 requests arrive simultaneously and none share a decode phase — the interleaving benefit requires at least some requests in decode while others are in prefill. With burst pattern this may not hold perfectly.

### Fast fallback

Drop `--enable-chunked-prefill`, keep `VLLM_ATTENTION_BACKEND=FLASHINFER --enable-prefix-caching`. Should still yield 2-3x over default.

---

## Sequencing Recommendation (1 GPU, 3 Students, ~2 hr window)

GPU is shared sequentially. Each server start + warmup + benchmark run takes ~30-40 min per scenario arm.

**Recommended order:**

| Slot | Time | Student | Hypothesis | Scenario | Primary Metric |
|------|------|---------|------------|----------|----------------|
| 1 | 0-40 min | Student 1 | H1: vLLM N-gram Spec Decode | B | TPOT speedup > 15.23x |
| 2 | 40-80 min | Student 2 | H2: vLLM FLASHINFER + FP8 | A | TTFT speedup > 4.37x |
| 3 | 80-120 min | Student 3 | H3: SGLang LPM | C | req/s > 51.12x |

**If time permits (2hr+ window):**
- Slot 4: H4 (TGI FP8 + CUDA Graphs, Scenario D) — most speculative, test last
- Slot 5: H5 (vLLM chunked prefill, Scenario D) — ablation vs H4

**Priority rationale:**
- H1 (Scenario B) has the largest absolute gap (2.25x default vs 15.23x SMAC3 ceiling). If ngram spec decode hits even 50% of the SMAC3 number, it is a major result.
- H2 (Scenario A) is high-confidence: FLASHINFER + FP8 are both well-validated on H100; risk is low.
- H3 (Scenario C) is already above SMAC3 with default SGLang; LPM is a small tuning bet, not a heroic swing. Lower priority if slots are scarce.
- H4 and H5 are more speculative (different engine, TGI install uncertainty).

---

## Quality Gate Reminder

Every hypothesis must pass: MMLU-Pro observed accuracy >= 0.95 * pytorch_fp16_baseline_accuracy.
Run quality eval before reporting results. FP8 configs are the main risk — test H2 and H4 quality first.
