# SENPAI Research State

- Updated: 2026-05-24 17:15Z
- Latest direction from human research team: multiple URGENT live notes on
  PR comments (16:29Z, 16:34Z) — **FLASHINFER and `--kv-cache-dtype fp8` are
  failed paths on this RTX PRO 6000 pod**. Use FLASH_ATTN backend and default
  KV dtype. `runtime_env.sh` now sets `VLLM_USE_FLASHINFER_SAMPLER=0` and
  `VLLM_DISABLE_FLASHINFER_PREFILL=1` defaults. `INFERENCE_BENCH_MAX_MODEL_LEN`
  defaults to 32768 (vLLM 0.11 rejects 131072).
- Active research tag: `ib-20260524-hardened-r3`, advisor branch
  `ib-20260524-hardened-r3`, target base `codex/inferencebench-senpai-target`.
- Hardware: 1x RTX PRO 6000 Blackwell, ~96GB VRAM (shakedown — not
  leaderboard-comparable).

## Current research focus

InferenceBench launcher search for Mistral-7B-Instruct-v0.3 in the 2 hour
SENPAI window. Each PR optimizes ONE scenario at a time and reports the
matching `scenario/<X>/speedup_over_pytorch` against the live PyTorch baseline
captured in scoring assets `rtxpro6000-seed248`.

Three logical students share a single GPU. Coordination via
`senpai/gpu_slot.py`. Heavy launches are serialized; prep work, log analysis,
and quick smoke tests can run in parallel. PR comments are for human-readable
coordination; the slot file is the source of truth for "who is allowed to
run a full eval right now".

## Round 1 hypotheses (revised after FLASHINFER/fp8 ban)

Round 1 targets the three highest-leverage scenarios per the public
reference snapshot. **Revised** after the human team's runtime notes: all
three launchers must use `FLASH_ATTN` backend and default KV dtype (no fp8
KV) plus `--tokenizer-mode mistral`. The hypotheses themselves are
unchanged — n-gram speculative decoding, large batching+prefix caching for
throughput, and chunked-prefill scheduler tuning are still valid; just the
backend/quant pieces had to be dropped.

| Student      | PR  | Scenario | Status            | Strategy                                                                                  |
|--------------|-----|----------|-------------------|-------------------------------------------------------------------------------------------|
| r3-frieren   | #66 | C        | WIP (orphan reap) | vLLM throughput-first: big batch, prefix cache, chunked prefill, FLASH_ATTN, default KV   |
| r3-fern      | #67 | B        | WIP               | vLLM decode-first: n-gram speculative decoding (num_speculative_tokens=5), FLASH_ATTN     |
| r3-tanjiro   | #68 | A        | WIP (pivoted)     | vLLM prefill-first: FLASH_ATTN + chunked prefill tuned for long-input burst c=1           |

Operational events worth remembering:
- An orphan vLLM process tree from a failed Sc.A attempt was holding ~90
  GiB GPU memory and blocking new launches. r3-frieren was authorized to
  reap it (one-time exception scoped to PID 14365 and its process group).
- r3-tanjiro hot-patched the pod with `cbor2 pyzmq nvidia-cuda-runtime-cu12
  nvidia-cublas-cu12 nvidia-cudnn-cu12 "transformers<5.0"` after vLLM 0.11.0
  failed to import on a stale image. Useful baseline for the next launch.

## Potential next research directions (after Round 1)

1. **Scenario D balanced launchers** — once A/B/C have measured baselines,
   try a single Mistral-7B launcher that scores well on D's geomean.
2. **Engine ablations** — re-run the same scenario with SGLang and TGI to
   confirm vLLM is the right base before pursuing custom kernels.
3. **Quantization variants** — beyond FP8 KV cache: weight FP8, AWQ, GPTQ
   on Mistral-7B. Check that MMLU-Pro stays above 0.2831.
4. **Speculative decoding variants** — n-gram vs draft-model vs EAGLE-style
   on Scenario B; tune `num_speculative_tokens` and lookahead.
5. **Attention backend sweep** — FLASH_ATTN vs FLASHINFER vs TRITON_ATTN on
   each scenario with otherwise identical args, to isolate the kernel win.
6. **Chunked prefill / scheduler interplay** — sweep
   `max-num-batched-tokens` alongside `max-num-seqs` for Sc. A and D.
7. **Prefix caching ROI** — measure the prefix-caching delta on Sc. C and D
   under the actual benchmark prompt distribution.
8. **`gpu-memory-utilization` ceiling** — push to 0.92-0.95 with FP8 KV
   cache to expand KV pool without OOM on the 96GB Blackwell.

## Notes

- Reference snapshot is H100; treat it as a ceiling indicator only on RTX
  PRO 6000. We are looking for the right serving recipe, not a numeric
  match.
- Quality gate at `tau=0.95` of PyTorch MMLU-Pro accuracy (0.298) is a hard
  constraint. Any candidate that fails quality is invalid regardless of
  speed.
