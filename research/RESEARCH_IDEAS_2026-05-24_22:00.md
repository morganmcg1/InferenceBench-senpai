# InferenceBench Round 1 Hypotheses — 2026-05-24

Base model: `mistralai/Mistral-7B-Instruct-v0.3`
Hardware: RTX PRO 6000 Blackwell, ~96 GB VRAM
Priority order: B > A > D (skip C — vLLM default already wins)

PyTorch baselines (speedup targets):
- A (ttft.p50): baseline=0.4385s; vLLM default H100=1.25x; SMAC3 H100 best=4.37x
- B (tpot.p50): baseline=0.025153s; vLLM default H100=2.25x; SMAC3 H100 best=15.23x
- D (geomean): vLLM default H100=1.96x; SMAC3 H100 best=5.69x

---

## Scenario B — Decode-Heavy (1024 in / 8192 out, concurrency 1)

### B1 — n-gram Speculative Decoding, 5 tokens (CREATIVE / NON-OBVIOUS)

**Mechanistic effect:** n-gram speculative decoding proposes the next 5 tokens by matching recent output in a sliding window of previous tokens, then verifies all 5 in a single forward pass. For decode-heavy workloads at concurrency=1 the GPU is mostly idle between decode steps; speculation fills that idle time with verified tokens and can multiply effective decode throughput by 3-6x with no draft model and no quality loss.

**Key flags / env vars (copy-paste into start_server.sh):**

```bash
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --disable-log-stats \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":1}' \
    --max-num-seqs 32 \
    --block-size 16
```

**Fallback if regression:** Drop `num_speculative_tokens` to 3 — fewer speculative tokens means fewer wasted verification passes when the n-gram match rate is low (e.g. highly varied outputs). 3-token speculation is robust across most output distributions.

---

### B2 — FP8 KV Cache + High GPU Utilization

**Mechanistic effect:** Quantising the KV cache from bf16 to FP8 halves its memory footprint. For an 8192-token decode at concurrency=1 the KV cache is the dominant memory consumer; freeing that memory allows vLLM to allocate more KV blocks and avoid cache evictions / re-computation, reducing per-token decode latency. No model weight quantisation — full bf16 inference quality is preserved.

**Key flags / env vars:**

```bash
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.94 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --disable-log-stats \
    --max-num-seqs 16 \
    --block-size 32
```

**Fallback if regression:** Revert `--kv-cache-dtype` to `auto`; keep `--gpu-memory-utilization 0.94` and `--block-size 32` to test whether the memory headroom alone (without FP8 rounding) explains any gain or loss.

---

## Scenario A — Prefill-Heavy (8192 in / 1024 out, concurrency 1)

### A1 — Chunked Prefill + Maximum Batched Tokens

**Mechanistic effect:** With an 8192-token input and concurrency=1, the entire prefill is a single large kernel that saturates the GPU compute but stalls the pipeline until complete. Enabling chunked prefill with a large `max_num_batched_tokens` allows vLLM to break the prefill into sub-chunks and overlap memory transfers with computation, reducing TTFT. Setting `max_num_batched_tokens=16384` (larger than the input) keeps the full prefill in one chunk but unlocks the chunked-prefill scheduler path which uses more efficient kernel dispatch on Blackwell.

**Key flags / env vars:**

```bash
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --disable-log-stats \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 16 \
    --block-size 16
```

**Fallback if regression:** Disable `--enable-chunked-prefill` and instead try `--max-num-batched-tokens 8192` without chunking — this tests whether the dispatcher path or the chunk size itself is the active lever.

---

### A2 — FP8 KV Cache for Prefill Memory Relief

**Mechanistic effect:** An 8192-token prefill fills the KV cache rapidly. FP8 KV cache halves the per-block KV memory cost, allowing more KV blocks to be pre-allocated before the prefill begins and reducing the chance of mid-prefill eviction stalls. Unlike scenario B, the benefit here is earlier block availability at the start of the request, directly shortening TTFT.

**Key flags / env vars:**

```bash
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --disable-log-stats \
    --max-num-seqs 16 \
    --max-num-batched-tokens 8192 \
    --block-size 16
```

**Fallback if regression:** If FP8 KV rounding hurts MMLU-Pro accuracy past the 0.95x gate, revert to `--kv-cache-dtype auto` and instead push `--gpu-memory-utilization 0.95` alone — this tests whether the accuracy hit or the memory savings is the dominant effect.

---

## Scenario D — Balanced (4096 in / 2048 out, concurrency 4)

### D1 — Tuned Concurrency + Chunked Prefill for Mixed Workload

**Mechanistic effect:** At concurrency=4 with mixed prefill/decode, the bottleneck shifts between prefill saturation and decode throughput request-by-request. Chunked prefill with moderate chunk size (4096) allows the scheduler to interleave decode steps for already-prefilled requests while a new request is still prefilling, reducing head-of-line blocking. `max_num_seqs=64` ensures enough slots to keep all 4 concurrent requests active without queuing.

**Key flags / env vars:**

```bash
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --disable-log-stats \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 64 \
    --block-size 16
```

**Fallback if regression:** Increase `--max-num-batched-tokens` to 8192 (fewer prefill chunks, less interleaving overhead) and drop `--max-num-seqs` to 32 — this tests whether the interleaving benefit or the concurrency slot count is the active lever.

---

### D2 — n-gram Speculative Decoding for Balanced Scenario (3 tokens, concurrency-safe)

**Mechanistic effect:** At concurrency=4 each sequence independently benefits from speculative decoding — n-gram proposals are per-sequence and do not interact across batch slots. The 2048-token output side of scenario D still has meaningful decode depth, and 3-token speculation is conservative enough to maintain high acceptance rates with mixed output content. Lowering speculation from 5 to 3 tokens reduces the wasted verify cost when batched verification of 4 sequences simultaneously fails, keeping the average acceptance overhead bounded.

**Key flags / env vars:**

```bash
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --disable-log-stats \
    --speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4,"prompt_lookup_min":1}' \
    --max-num-seqs 64 \
    --block-size 16
```

**Fallback if regression:** Disable speculative decoding entirely and instead test `--max-num-seqs 128` with `--max-num-batched-tokens 8192` — this isolates whether speculative overhead at concurrency=4 is net negative vs. the batch sizing being the real bottleneck.

---

## Summary Table

| ID | Scenario | Primary Lever | Expected Gain Mechanism |
|----|----------|--------------|------------------------|
| B1 | B (tpot) | n-gram spec decode, 5 tok | Fills idle decode GPU time; 3-6x effective token rate |
| B2 | B (tpot) | FP8 KV cache | Halves KV memory, more blocks, fewer evictions |
| A1 | A (ttft) | Chunked prefill + 16384 batched tok | Efficient kernel dispatch for large prefill on Blackwell |
| A2 | A (ttft) | FP8 KV cache + 0.95 GMU | More pre-allocated KV blocks before prefill begins |
| D1 | D (geomean) | Chunked prefill + 4096 chunks + 64 seqs | Interleaves decode/prefill, reduces HOL blocking |
| D2 | D (geomean) | n-gram spec decode, 3 tok | Per-sequence speculation at batch depth 4 |

Creative/non-obvious pick: **B1** — n-gram speculative decoding requires no draft model, adds zero training cost, exploits the token repetition structure inherent in long assistant-style outputs (lists, code, templates), and is the single highest-leverage lever for decode-heavy tpot at concurrency=1.
