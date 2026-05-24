# Research Ideas — InferenceBench SENPAI, 2026-05-24

**Context:** RTX PRO 6000 Blackwell, ~96 GB VRAM, Mistral-7B-Instruct-v0.3,
2-hour budget per scenario. All four scenarios are tabula rasa on this branch —
no SENPAI launcher has been measured yet. H100 reference: SMAC3 best is 11.53×
aggregate (Sc.A 4.37×, Sc.B 15.23×, Sc.C 46.70×, Sc.D 5.69×).

Quality gate: MMLU-Pro >= 0.95× PyTorch baseline. Paper-facing metric: speedup
over PyTorch baseline. FlashInfer sampler and prefill are DISABLED by default
in `runtime_env.sh`; any hypothesis that re-enables them must do so explicitly.

Hypotheses are ranked by expected speedup × success probability. Each includes
a concrete `start_server.sh` skeleton — copy, adjust paths, and wire `HOST`,
`PORT`, and `MODEL_ID` from environment variables before filing a student PR.

---

## Hypothesis 1 — vLLM FlashInfer Attention Backend (Sc. A + D, high priority)

**Rank: 1** | Expected uplift: 1.3–1.8× over vLLM default on TTFT | Confidence: high

### What it is

Re-enable the FlashInfer attention backend for vLLM on the RTX PRO 6000. The
default `runtime_env.sh` disables it (`VLLM_DISABLE_FLASHINFER_PREFILL=1`,
`VLLM_USE_FLASHINFER_SAMPLER=0`) as a shakedown-safety default. FlashInfer's
fused prefill kernel is specifically faster than FlashAttention for the
long-context (8K input) case that defines Scenario A.

### Mechanism

FlashInfer implements a page-table-aware grouped-query attention kernel that
fuses KV-cache index lookups into the CUDA kernel, eliminating scatter/gather
overhead that FlashAttention incurs. For Sc.A (8192-token inputs, 1024-token
outputs, c=1), the entire speedup is in prefill latency (TTFT). FlashInfer's
chunked-prefill kernel is measurably faster than FA2 for sequences >4K on
Ampere/Hopper hardware; on Blackwell, the same kernel compiles against the
sm_90 target and the Blackwell FP16 TMA path should provide additional benefit.

The vLLM FLASHINFER attention backend is selected at server start via the
`VLLM_ATTENTION_BACKEND=FLASHINFER` environment variable. Re-enabling the
sampler (`VLLM_USE_FLASHINFER_SAMPLER=1`) and prefill
(`VLLM_DISABLE_FLASHINFER_PREFILL=0`) is needed to activate it fully.

### Target scenario(s)

Primary: A (TTFT, long input). Secondary: D (balanced, mixed input/output).

### Concrete launcher config

```bash
#!/usr/bin/env bash
set -euo pipefail
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"

source "$PROBLEM_DIR/senpai/runtime_env.sh"

# Explicitly re-enable FlashInfer (overrides runtime_env.sh defaults)
export VLLM_ATTENTION_BACKEND="FLASHINFER"
export VLLM_USE_FLASHINFER_SAMPLER="1"
export VLLM_DISABLE_FLASHINFER_PREFILL="0"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.92 \
  --block-size 16 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --dtype bfloat16
```

### Risk / quality concern

FlashInfer has known issues with certain Blackwell PTX paths — if the vLLM
installed version predates Blackwell sm_100 support, the kernel falls back
silently to CUDA-core paths and may regress. Check server logs for
`Using FlashInfer backend` confirmation. Quality gate should be unaffected
(attention backend is numerically equivalent for FP16/BF16).

### Key references

- FlashInfer: "A Library for Efficient and Customizable Attention Acceleration"
  (Ye et al., 2024), https://arxiv.org/abs/2501.01005 — shows 1.3–2.1× TTFT
  speedup vs FlashAttention-2 on long-context prefill.
- vLLM FlashInfer integration: https://docs.vllm.ai/en/latest/features/flashinfer.html

---

## Hypothesis 2 — vLLM n-gram Speculative Decoding (Sc. B, high priority)

**Rank: 2** | Expected uplift: 1.5–2.5× on TPOT | Confidence: high

### What it is

Enable vLLM's prompt-lookup (n-gram) speculative decoding for Scenario B, which
has 8192-token outputs from 1024-token inputs. The generated tokens are the
dominant bottleneck; n-gram speculation can propose multiple tokens per step
from the draft vocabulary.

### Mechanism

Prompt-lookup decoding (PLDecoding, Fu et al. 2024) constructs draft tokens
by searching the prompt for n-gram matches to the current context. For
instruction-following tasks (Mistral-7B-Instruct-v0.3 on LongBench-v2), a
meaningful fraction of generated tokens mirror phrases in the prompt (quotation,
paraphrasing, named entities). Each accepted draft token reduces the number of
full-model forward passes. With 7 speculative tokens and an acceptance rate of
~50%, TPOT can halve because two tokens are produced per verify-pass instead of
one. The RTX PRO 6000 has 96 GB VRAM — no memory pressure from a draft model.

### Target scenario(s)

Primary: B (TPOT, long output). The H100 reference shows Sc.B is the highest-
leverage scenario (SMAC3 achieves 15.23× vs 4.37× for Sc.A).

### Concrete launcher config

```bash
#!/usr/bin/env bash
set -euo pipefail
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"

source "$PROBLEM_DIR/senpai/runtime_env.sh"

# n-gram (prompt-lookup) speculative decoding
# --speculative-config sets num_speculative_tokens when engine is vLLM 0.11+
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --block-size 16 \
  --speculative-config '{
    "method": "ngram",
    "num_speculative_tokens": 7,
    "prompt_lookup_min": 2,
    "prompt_lookup_max": 4
  }' \
  --dtype bfloat16
```

Note: In vLLM >=0.11, the `--speculative-config` flag takes a JSON dict
specifying the speculative method. If the installed version uses the older
`--num-speculative-tokens` flag with `--speculative-draft-tensor-parallel-size`,
use that form instead. Check `python -m vllm.entrypoints.openai.api_server --help`
on the pod before committing to the flag format. The search space YAML uses
`num_speculative_tokens` [0,3,5,7] — 7 is the upper bound to try first.

### Risk / quality concern

n-gram speculative decoding is draft-free (no auxiliary model), so quality gate
impact is minimal — the verify pass uses the full model and rejects incorrect
drafts. The main failure mode is low acceptance rate on this dataset, which
would give overhead with no speedup. Run a quick test with `--quick` eval first.

### Key references

- "Prompt Lookup Decoding" (Fu et al., 2024),
  https://github.com/apoorvumang/prompt-lookup-decoding — original method.
- "SpecBench" (Xia et al., 2024), https://arxiv.org/abs/2401.07851 — finds
  1.7–2.7× TPOT speedup on instruction tuned models vs no spec decoding.
- vLLM speculative decoding docs:
  https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/

---

## Hypothesis 3 — SGLang FP8 Weight Quantization + RadixAttention (Sc. C, high priority)

**Rank: 3** | Expected uplift: 1.3–1.7× on throughput | Confidence: medium-high

### What it is

Run Scenario C (high-load throughput) on SGLang with FP8 weight quantization
enabled. The SGLang search space YAML currently has `quantization: ["none"]` —
FP8 is an unexplored lever. SGLang's RadixAttention prefix cache should
compound with quantization to give more KV cache space per token.

### Mechanism

FP8 weight quantization (W8A16 or W8A8 depending on backend) halves the weight
memory footprint from ~14 GB (BF16 7B) to ~7 GB, freeing ~7 GB VRAM for KV
cache. On Scenario C (256 req × 3 profiles, concurrency up to 64), more KV
cache space directly increases the maximum concurrent batch size before eviction.
SGLang's RadixAttention (radix-tree prefix cache) also reuses shared prefill
chunks across concurrent requests — on LongBench-v2 prompts where the system
prompt is shared, this can deliver significant prefill reuse. Combined, the two
effects target throughput, not latency.

SGLang uses `--quantization fp8` (or `--quantization marlin` for GPTQ-Marlin).
The RTX PRO 6000 Blackwell SM89+ supports FP8 GEMM natively.

### Target scenario(s)

Primary: C (req/s throughput). Secondary: D (balanced).

### Concrete launcher config

```bash
#!/usr/bin/env bash
set -euo pipefail
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"

source "$PROBLEM_DIR/senpai/runtime_env.sh"

exec python -m sglang.launch_server \
  --model-path "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len 32768 \
  --max-running-requests 128 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.88 \
  --quantization fp8 \
  --schedule-policy lpm \
  --dtype bfloat16
```

Note: SGLang's `--quantization fp8` performs online per-tensor FP8 quantization
of weights at load time. If `fp8` is not supported in the installed SGLang
version, try `--quantization w8a16` (Marlin-style) or check
`python -m sglang.launch_server --help`. The search space YAML restriction to
`quantization: ["none"]` reflects a conservative baseline choice, not a
technical limitation.

### Risk / quality concern

FP8 per-tensor quantization can degrade MMLU-Pro by 1–3%. KVQuant research
shows <0.1 perplexity degradation on Mistral at 3-bit KV; FP8 weight-only
should be safer. Run quality gate evaluation before reporting speedup. If MMLU-
Pro drops below 0.95× baseline, try `--quantization none` with only the batching
tweaks as a fallback.

### Key references

- SGLang (Zheng et al., NeurIPS 2024): RadixAttention delivers up to 6.4×
  throughput vs prior SOTA, https://arxiv.org/abs/2312.07104
- KVQuant (Hooper et al., 2024): FP8 / low-bit KV quantization on Mistral
  with <0.1 perplexity loss, https://arxiv.org/abs/2401.18079

---

## Hypothesis 4 — vLLM FP8 KV Cache + Chunked Prefill (Sc. A + C, medium priority)

**Rank: 4** | Expected uplift: 1.2–1.5× | Confidence: medium

### What it is

Enable FP8 KV cache dtype in vLLM alongside chunked prefill for scenarios A
and C. FP8 KV cache halves the per-token KV footprint, allowing a larger
effective batch size before paging. Chunked prefill interleaves prefill and
decode to reduce head-of-line blocking on TTFT.

### Mechanism

Scenario A uses 8192-token inputs at concurrency 1 — the KV cache for a single
request spans 8192 × 32 layers × 32 heads × 128 dim × 2 (K,V) × 2 bytes (BF16)
≈ 4.3 GB. FP8 halves this to ~2.2 GB, enabling the server to hold more
concurrent requests in memory. For Sc.A at c=1 this does not change parallelism,
but it directly reduces prefill memory bandwidth pressure which can improve TTFT.
For Sc.C at c=64, the freed KV memory directly increases sustainable batch size.

Chunked prefill (`--enable-chunked-prefill`) splits long prefills into chunks
(default chunk = `max-num-batched-tokens`), allowing the scheduler to interleave
decode steps. For Sc.A this reduces stall on the decode of previously-issued
requests, but with c=1 the main benefit is on the prefill compute efficiency.

### Target scenario(s)

Primary: C (throughput uplift from larger batch). Secondary: A (TTFT reduction
from lower KV memory pressure and chunked prefill interleaving).

### Concrete launcher config

```bash
#!/usr/bin/env bash
set -euo pipefail
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"

source "$PROBLEM_DIR/senpai/runtime_env.sh"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.93 \
  --block-size 16 \
  --kv-cache-dtype fp8 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --dtype bfloat16
```

### Risk / quality concern

vLLM FP8 KV cache uses per-tensor static scaling that may degrade quality more
than per-channel FP8. On Mistral-7B (GQA, 8 KV heads), the KV tensors are small
per head and per-tensor scaling is coarser. Run the quality gate check carefully.
If MMLU-Pro fails, try `--kv-cache-dtype auto` (BF16) as a fallback — chunked
prefill alone still provides value.

### Key references

- "No Token Left Behind: Reliable KV Cache Compression via Importance-Aware
  Mixed Precision Quantization" (He et al., 2024),
  https://arxiv.org/abs/2402.18096 — shows FP8 KV cache is safe on 7B models.
- vLLM chunked prefill docs:
  https://docs.vllm.ai/en/latest/features/chunked_prefill.html

---

## Hypothesis 5 — SGLang FlashInfer Attention + CUDA Graphs (Sc. B + D, medium priority)

**Rank: 5** | Expected uplift: 1.2–1.5× on TPOT | Confidence: medium

### What it is

SGLang's search space YAML only lists `attention_backend: ["triton"]`. FlashInfer
and CUDA graphs are also available in recent SGLang builds but are not exercised
by the baseline HPO. This hypothesis enables both for Scenario B (long decode)
and D (balanced).

### Mechanism

SGLang 0.4+ supports `--attention-backend flashinfer` and
`--cuda-graph-max-bs <N>`. For decode-dominated workloads (Sc.B: 8192-token
outputs), CUDA graph capture eliminates CPU kernel launch latency per token step.
At batch size ≤ 32, the overhead of Python/CUDA API calls is proportionally
large relative to the short compute kernel. CUDA graph capture records the GPU
command sequence once and replays it without CPU re-dispatch — a known
2–4 ms/token saving at small batch sizes.

FlashInfer's decode kernel (`BatchDecodeWithPagedKV`) is also more cache-
friendly than Triton's paged decode kernel at the small batch sizes common in
Sc.B (c=1, sequential requests).

### Target scenario(s)

Primary: B (TPOT). Secondary: D (balanced).

### Concrete launcher config

```bash
#!/usr/bin/env bash
set -euo pipefail
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"

source "$PROBLEM_DIR/senpai/runtime_env.sh"

exec python -m sglang.launch_server \
  --model-path "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len 32768 \
  --max-running-requests 32 \
  --chunked-prefill-size 4096 \
  --mem-fraction-static 0.88 \
  --attention-backend flashinfer \
  --cuda-graph-max-bs 32 \
  --schedule-policy fcfs \
  --dtype bfloat16
```

### Risk / quality concern

SGLang FlashInfer support on Blackwell (sm_100) may require a recent nightly
build. Check with `python -c "import flashinfer; print(flashinfer.__version__)"`.
If FlashInfer is not installed or fails to initialize, SGLang falls back to
Triton without error but logs a warning. CUDA graph capture at bs=32 uses ~1 GB
extra VRAM for the replay buffers — not a concern on 96 GB.

### Key references

- FlashInfer (Ye et al., 2024), https://arxiv.org/abs/2501.01005 — shows
  decode throughput gains of 1.3–2.1× vs Triton at small batch sizes.
- SGLang CUDA graph docs:
  https://docs.sglang.io/backend/server_arguments.html#cuda-graph

---

## Hypothesis 6 — TGI FP8 Quantization + Large Prefill Batch (Sc. C, medium priority)

**Rank: 6** | Expected uplift: 1.1–1.4× throughput | Confidence: medium

### What it is

HuggingFace TGI's search space already includes `quantize: fp8`, `eetq`, and
`awq`. The baseline HPO may not have converged on the best combination of FP8
quantization + large `max_batch_total_tokens` + tuned `max_batch_prefill_tokens`.
This hypothesis targets a hand-crafted TGI config for Sc.C throughput.

### Mechanism

TGI with `--quantize fp8` uses the `eetq` FP8 path on Blackwell hardware
(SM89+), which activates native FP8 GEMM. The freed weight memory is reallocated
to KV cache. For Sc.C (burst-c64), a larger `max_batch_total_tokens` allows TGI
to batch more decode steps per forward pass, and `max_batch_prefill_tokens` at
16384 allows full 16K-token prefill chunks to be processed in a single step
rather than split across iterations.

TGI's continuous batching with `waiting_served_ratio=1.2` (the balanced default)
allows waiting requests to join once the ratio of waiting-to-serving is exceeded,
reducing head-of-line blocking at high concurrency.

### Target scenario(s)

Primary: C (throughput). TGI's H100 default score (41.94 req/s) is below SGLang
and vLLM defaults, so this is an upside scenario if TGI's FP8 path is well-
optimized for Blackwell.

### Concrete launcher config

```bash
#!/usr/bin/env bash
set -euo pipefail
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"

source "$PROBLEM_DIR/senpai/runtime_env.sh"

exec text-generation-launcher \
  --model-id "$MODEL_ID" \
  --hostname "$HOST" \
  --port "$PORT" \
  --max-input-length 32768 \
  --max-total-tokens 33792 \
  --max-concurrent-requests 128 \
  --max-batch-prefill-tokens 16384 \
  --max-batch-total-tokens 131072 \
  --max-waiting-tokens 100 \
  --waiting-served-ratio 1.2 \
  --quantize fp8 \
  --cuda-graphs "1,2,4,8,16,32" \
  --cuda-memory-fraction 0.93
```

### Risk / quality concern

TGI's `--quantize fp8` behaviour differs by version. In older TGI (<2.0), `fp8`
maps to `eetq` dynamic FP8; in newer TGI, it may use a different kernel path.
Check `text-generation-launcher --version` and TGI changelog. Quality gate risk
is moderate — validate MMLU-Pro before reporting. If it fails, fall back to
`--quantize eetq` which uses dynamic scaling (safer quality) or `--quantize none`.

### Key references

- TGI FP8 docs: https://huggingface.co/docs/text-generation-inference/conceptual/quantization
- EETQ: https://github.com/NetEase-FuXi/EETQ — INT8/FP8 quantization library
  used by TGI, showing 1.2–1.5× throughput on 7B models.

---

## Hypothesis 7 — SGLang LPM Scheduling + Prefix Caching (Sc. C + D, medium priority)

**Rank: 7** | Expected uplift: 1.1–1.3× | Confidence: medium-low

### What it is

SGLang's search space includes `schedule_policy: ["fcfs", "lpm"]` but the
baseline HPO may not isolate LPM's value. LPM (Longest Prefix Match) is SGLang's
RadixAttention-aware scheduler that prioritizes requests with the longest cache
hit — compounding the prefix cache benefit. This hypothesis pairs LPM scheduling
explicitly with high `mem-fraction-static` to maximize cache size.

### Mechanism

LPM scheduling (implemented in SGLang as of 0.3) routes incoming requests to
processes that already hold a prefix match in their RadixAttention cache. For
Sc.C (LongBench-v2, 1K-token inputs, 256 requests, 3 profiles), many requests
share a common system prompt or preamble — LPM prioritizes those requests to
maximize cache reuse, reducing effective prefill cost per request. k-LPM
(Podder et al., NeurIPS 2025 workshop) shows 15–30% TTFT reduction vs FCFS in
prefix-heavy workloads.

### Target scenario(s)

Primary: C (throughput from reduced effective prefill cost). Secondary: D.

### Concrete launcher config

```bash
#!/usr/bin/env bash
set -euo pipefail
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"

source "$PROBLEM_DIR/senpai/runtime_env.sh"

exec python -m sglang.launch_server \
  --model-path "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len 32768 \
  --max-running-requests 256 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.90 \
  --attention-backend triton \
  --schedule-policy lpm \
  --dtype bfloat16
```

### Risk / quality concern

LPM's benefit depends on prefix reuse in the benchmark workload. If LongBench-v2
requests in Sc.C have low prefix overlap (each request is a distinct long-context
query), LPM degrades to FCFS with overhead. Inspect the request file with
`python senpai/materialize_requests.py --scenario C --dry-run` if available to
estimate prefix overlap before committing full evaluation time.

### Key references

- SGLang RadixAttention (Zheng et al., NeurIPS 2024),
  https://arxiv.org/abs/2312.07104 — LPM scheduling is Section 4.2.
- k-LPM (Podder et al., 2025) — shows 15–30% TTFT reduction in prefix-reuse
  settings vs FCFS at matched throughput.

---

## Hypothesis 8 — vLLM Prefix Caching + Large Block Size (Sc. D, lower priority)

**Rank: 8** | Expected uplift: 1.1–1.2× | Confidence: low-medium

### What it is

Enable prefix caching in vLLM with `block-size=32` (vs default 16) for Scenario
D. Larger block size reduces the number of KV cache index lookups and increases
cache line utilization per block — a small but free gain when prefix reuse is
present.

### Mechanism

Scenario D (4K input, 2K output, c=4) has moderate prefix length and burst
concurrency 4. With prefix caching enabled, requests sharing an instruction
prefix reuse cached KV blocks. Block size 32 means fewer block-boundary
bookkeeping operations per GQA attention call. The marginal gain is small (~5–
10%), but with no quality risk and negligible implementation complexity it is
worth profiling as a D-specific tune.

### Target scenario(s)

Primary: D (balanced geomean). This is a fine-tuning hypothesis, not a
transformational one — it should be attempted after higher-ranked ideas have
been measured.

### Concrete launcher config

```bash
#!/usr/bin/env bash
set -euo pipefail
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"

source "$PROBLEM_DIR/senpai/runtime_env.sh"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --block-size 32 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --dtype bfloat16
```

### Risk / quality concern

Prefix caching is numerically transparent (reuses exact cached tensors). No
quality risk. Low-confidence uplift — this is primarily a diagnostic to
understand whether D benefits from caching before adding more complexity.

---

## Prioritization Summary

| Rank | Hypothesis | Engine | Scenario | Mechanism | Expected uplift | Confidence |
|------|-----------|--------|----------|-----------|----------------|------------|
| 1 | FlashInfer attention backend | vLLM | A, D | Faster prefill kernel on Blackwell | 1.3–1.8× TTFT | High |
| 2 | n-gram speculative decoding | vLLM | B | Draft tokens from prompt, skip verify passes | 1.5–2.5× TPOT | High |
| 3 | SGLang FP8 + RadixAttention | SGLang | C, D | Free KV memory → larger batch | 1.3–1.7× req/s | Med-High |
| 4 | FP8 KV cache + chunked prefill | vLLM | A, C | Lower KV memory pressure | 1.2–1.5× | Medium |
| 5 | SGLang FlashInfer + CUDA graphs | SGLang | B, D | Kernel speed + graph replay | 1.2–1.5× TPOT | Medium |
| 6 | TGI FP8 + large prefill batch | TGI | C | Native FP8 GEMM + larger batch window | 1.1–1.4× req/s | Medium |
| 7 | SGLang LPM scheduling | SGLang | C, D | Prefix cache hit maximization | 1.1–1.3× | Med-Low |
| 8 | vLLM prefix caching + block32 | vLLM | D | Caching + reduced index overhead | 1.1–1.2× | Low-Med |

## Assignment recommendation for 3 students

- **Student 1:** Hypotheses 1 (FlashInfer/vLLM, Sc.A) — highest-confidence,
  single env-var change, direct diagnostic of the runtime_env.sh default.
- **Student 2:** Hypothesis 2 (n-gram speculative decoding, Sc.B) — Sc.B has
  the largest headroom on H100 (15.23× SMAC3) and spec decoding is the clearest
  mechanism for TPOT improvement.
- **Student 3:** Hypothesis 3 (SGLang FP8 + RadixAttention, Sc.C) — Sc.C is
  the throughput scenario, SGLang is the strongest throughput engine (51.12 req/s
  vs vLLM 48.69 on H100 default), and FP8 is an unexplored lever in the SGLang
  search space.

Once those three return results, prioritize Hypotheses 4 and 5 for the next
round depending on which scenario is furthest from the H100 SMAC3 reference.
