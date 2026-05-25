# SENPAI Research State

- **Date/time:** 2026-05-25 (start of ib-20260525-one3-r1 session)
- **Research tag:** ib-20260525-one3-r1
- **Hardware:** 1× RTX PRO 6000 Blackwell ~96GB VRAM (shakedown; leaderboard claims need H100)
- **Time budget:** 2 hours total from start gate

## Most recent research direction from human researcher team

No human-team directives received yet in this session. Proceeding per advisor
prompt contract.

## Current research focus and themes

**First PR in flight:** Scenario A prefill tuning (PR #100, student: frieren)

The opening play is to establish a clean vLLM-on-RTX-PRO-6000 baseline for
Scenario A (long-context input-heavy, 8K/1K, concurrency 1, burst TTFT.p50)
and immediately discover which prefill knobs move the needle. This is a fast
evaluating scenario (~12 min full eval) that gives us early evidence about how
aggressive our optimization budget should be.

### Themes in flight

1. **Prefill throughput (Scenario A):** max_num_batched_tokens, chunked prefill
   vs. monolithic, CUDA graph, memory utilization. These are the first-order
   levers for single-stream 8K-token prefill latency on vLLM.

2. **Engine health check:** First run also validates the stack — runtime_env.sh,
   FlashAttention backend (FlashInfer explicitly off per shakedown hardware),
   gpu_slot.py coordination, W&B logging pipeline, clean relaunch — before we
   push toward harder experiments.

## Potential next research directions and themes

Listed roughly by expected value given one GPU and ~90 min remaining after PR
#100 lands:

### Short-term (next 1–2 PRs)

- **Scenario B decode tuning** — TPOT.p50 at concurrency 1 with 8K output
  tokens. Biggest absolute headroom (default 2.25x, HPO 15.23x). Key lever:
  n-gram speculative decoding (`--speculative-config '{"method":"ngram",...}'`)
  because concurrency=1 decode is the ideal spec-dec setting. Also FP8 weight
  quantization to reduce memory and CUDA graph warmup.
- **Scenario A follow-up: Triton backend** — ablate `TRITON_ATTN` vs
  `FLASH_ATTN` on the winning Scenario A launcher to characterize attention
  backend on this specific Blackwell GPU.

### Medium-term

- **Scenario D balanced** — 4K/2K, concurrency 4, burst. The knobs that help A
  (prefill tuning) and B (decode, spec-dec) partially stack here. Worth a PR
  once we have A and B winners to reference.
- **FP8 weight quantization** — bitsandbytes or vLLM native FP8 weights to
  reduce memory pressure and increase batch capacity. Must clear quality gate.
- **SGLang for Scenario C** — vLLM defaults already dominate C (throughput)
  per the public reference (48.69x vs HPO 46.70x). SGLang has strong RadixAttention
  for prefix-cached workloads; check whether it beats vLLM default here when
  the workload has no natural shared prefix.

### Exploratory (if time allows)

- **TensorRT-LLM** — Potentially strong for Scenario A/B if the environment
  has a pre-compiled engine. Risky on time given compilation overhead.
- **Speculative decoding with target-as-draft** — Mistral 7B + n-gram prompts
  lookup for Scenario B long decode sequences.
- **Custom OpenAI-compatible server** — e.g. a minimal server that bypasses
  vLLM's tokenizer_mode overhead for scenarios where input token length is the
  hot path. Very exploratory.

## Key invariants to preserve

- All results must use the pre-staged scoring assets from PVC:
  `--import-dir /mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`
- Every terminal result needs a full `evaluate.py --json-output-file` run, a
  W&B log under `wandb-applied-ai-team/inferencebench-senpai`, and a clean
  relaunch confirm.
- RTX PRO 6000 shakedown: no FlashInfer unless explicitly tested; no FP8 KV
  cache unless a boot-to-quality run proves it works on this hardware.
- `BASELINE.md` is advisor-owned and updated only from verified terminal runs.
