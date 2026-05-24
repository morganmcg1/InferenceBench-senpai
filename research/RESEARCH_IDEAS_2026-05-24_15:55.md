# SENPAI Round 1 Research Ideas — ib-20260524-hardened-r5

Curated by the advisor at launch. The top three are the round-1 assignments;
the rest are queued for round 2+ depending on round-1 evidence.

## Round 1 — assigned to idle students

### 1. r5-frieren · Scenario B · `vllm-fp8-ngram-spec-scB`
- **Why now**: Scenario B has the biggest gap from vLLM default (2.25×) to the
  best public H100 result (15.23×) of any scenario, and the workload is
  literally 1024-in / 8192-out at concurrency 1, which is the ideal regime for
  prompt-lookup n-gram speculative decoding.
- **Stack**: vLLM + FP8 weight quant + FP8 KV cache + n-gram speculative decoding
  with `num_speculative_tokens=5`, `prompt_lookup_max=5` + FlashInfer attention
  + CUDA graphs (`enforce_eager=False`).
- **Risk**: Quality gate. FP8 + speculative decoding can drift MMLU-Pro
  slightly; instruct the student to verify quality and to fall back to bf16
  weights with FP8 KV + spec if the gate fails.

### 2. r5-fern · Scenario A · `vllm-fp8-flashinfer-prefill-scA`
- **Why now**: Scenario A is 8192 input / 1024 output at concurrency 1 — the
  TTFT is dominated by the long prefill, and FlashInfer's prefill kernel plus
  FP8 weights should compress that step heavily versus the default
  FLASH_ATTN path.
- **Stack**: vLLM + FP8 + FP8 KV cache + FLASHINFER attention backend +
  `max_num_batched_tokens=16384` so the 8192-token prefill is unsplit +
  `enable_chunked_prefill=False` (no benefit at concurrency 1) + CUDA graphs.
- **Risk**: FlashInfer wheel stability on Blackwell. If FLASHINFER refuses to
  load, fall back to FLASH_ATTN with the same FP8 stack.

### 3. r5-tanjiro · Scenario D · `vllm-fp8-spec-balanced-scD`
- **Why now**: Scenario D is the general balanced workload (4096-in / 2048-out,
  burst c=4) and its score is the geomean of TTFT, TPOT, and req/s, so it
  rewards a configuration that wins everywhere instead of one extreme. FP8 +
  n-gram speculative decoding is exactly that kind of horizontally good stack.
- **Stack**: vLLM + FP8 + FP8 KV cache + n-gram speculative (k=5) + chunked
  prefill on with `max_num_batched_tokens=8192` + `max_num_seqs=64` +
  FlashInfer attention + CUDA graphs.
- **Risk**: Speculative decoding can hurt batched decode at higher concurrency
  if acceptance rates are low. Have the student report acceptance rate; if
  poor, drop spec and rerun.

## Round 2 — queued

- **AWQ INT4 or GPTQ INT4 Mistral-7B-Instruct-v0.3** on the winning scenario.
  Often beats FP8 on Blackwell decode workloads.
- **TensorRT-LLM** head-to-head against the round-1 winner on B or D.
- **SGLang with FA3 backend** on the winning scenario.
- **EAGLE / Medusa speculative decoding** on B (needs a draft model).
- **Prefix-cache micro-sweep on C** with short-shared system prompts.
- **Chunked-prefill sweep on A** — small chunks vs no chunks at conc=1 with
  long context (sometimes scheduler delays hide the prefill win).
- **KV-cache block size sweep on C** (16 vs 32) — the leaderboard is so close
  that small alloc changes may flip rankings.
- **Compile + custom Triton attention** for Mistral specifically on Blackwell.

## Notes on coordination

- All three round-1 assignments target the same GPU pod. Each PR instructs the
  student to grab `senpai/gpu_slot.py` before any heavy run and release it
  with `SLOT-FREE` afterwards, so we never run two servers concurrently.
- Priority order if the slot queue stalls: B > D > A. B has the largest
  speedup leverage and the cheapest validation (only 64 requests on the burst
  profile), so it should run first.
