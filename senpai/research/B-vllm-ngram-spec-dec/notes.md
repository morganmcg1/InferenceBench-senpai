# Scenario B — vLLM n-gram Speculative Decoding

First measured candidate for Scenario B on `ib-20260525-one3-r1`.

## Levers committed in this launcher (`senpai/launchers/B/vllm-ngram-spec-dec/start_server.sh`)

- Engine: vLLM (`vllm.entrypoints.openai.api_server`, v0.11.0).
- Attention backend: FlashAttention (`VLLM_ATTENTION_BACKEND=FLASH_ATTN`); FlashInfer prefill stays disabled via `senpai/runtime_env.sh` defaults.
- **n-gram speculative decoding** via `--speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'`. Each decode step proposes up to 5 tokens from prompt n-gram matches and verifies them in a single forward pass.
- `--max-num-seqs 2`: covers the draft+verification slot pair at concurrency 1.
- `--max-num-batched-tokens 2048`: comfortable for the 6-token verification step; avoids prefill chunking interactions.
- `--block-size 32`: larger KV blocks reduce management overhead at 8K decode.
- `--no-enable-chunked-prefill` is requested in the CLI; vLLM still reports `chunked_prefill_enabled=True` because spec dec engages the V1 unified scheduler, but the 1K prefill is still served in a single forward pass at concurrency 1.
- `--max-model-len 10240`: KV pool just larger than 1K prefill + 8K decode.
- `--gpu-memory-utilization 0.90`; BF16 KV cache; CUDA graphs ON; `--no-enable-prefix-caching`; `--seed 248`.

## Single-arm eval (no sweep)

The PR brief allotted **one arm**: 5 spec tokens, 2–4 gram lookup. No alternative spec-token count was tried — quick eval confirmed boot + huge gain, so full eval ran directly.

## Quick eval (4-request burst, 16-question MMLU-Pro)

- TPOT.p50 = 0.00737 s → ~3.42× speedup vs PyTorch (0.0252 s)
- TTFT.p50 = 0.0639 s; gen tok/s = 124.96
- MMLU-Pro 16q: 0.1875 (ratio 0.629, FAIL) — sample noise at n=16; PyTorch baseline at n=16 sits in the same band (95% CI ≈ 0.07–0.53 for true 0.298).
- Decision: proceed straight to full eval.

## Full-eval result (W&B run `kff5nbhz`)

| Metric | Candidate | PyTorch | Speedup |
|---|---:|---:|---:|
| **TPOT.p50** | **0.00839 s** | 0.02515 s | **2.997×** |
| TTFT.p50 | 0.0634 s | 0.0709 s | 1.117× |
| ITL.p50 | 0.01302 s | 0.01462 s | — |
| gen tok/s (burst) | 116.61 | 39.19 | 2.975× |
| req/s (burst) | 0.02396 | 0.01332 | 1.799× |
| success / fail / empty | 64 / 0 / 0 | 64 / 0 / 0 | — |

- **Quality:** MMLU-Pro observed 0.302 vs baseline 0.298 → ratio 1.0134, **gate PASS** (τ=0.95, n=500). Spec dec preserved (or marginally improved) quality, consistent with vLLM's verified rejection sampling being mathematically equivalent to non-spec sampling.
- VRAM peak: 88245 MiB / ~96 GiB on RTX PRO 6000.
- Clean relaunch: launcher booted via `test_server.sh` inside the supervised `gpu_slot.py run` group; the same `start_server.sh` cold-booted from a fresh shell to serve full eval.
- W&B: `wandb-applied-ai-team/inferencebench-senpai/runs/kff5nbhz`, group `ib-20260525-one3-r1-B-vllm-ngram-spec-dec`.

## Comparison to BASELINE.md

- Scenario B row was previously empty (no measured candidate). This is the first.
- Public reference (H100, vLLM default, no agent) reports 2.25× on B; RTX PRO 6000 vLLM-default has not been measured but is generally close to that number. The 2.997× from spec dec is therefore a clear gain over vLLM-default at this scenario, and lands well above the table's "vLLM default" entry.
- Headroom to public best on H100 (15.23×): substantial, but those numbers come from SMAC3 HPO search across many flags and from a different GPU class. On RTX PRO 6000 the achievable ceiling is unknown.

## What worked

- n-gram speculative decoding is an excellent match for output-heavy long-decode workloads at concurrency 1. With `ignore_eos=True` and 8K output tokens, the 8192-token horizon dominates wall time; cutting per-token decode latency 3× was the right knob.
- Quality is fully preserved — actually slightly better in sample (0.302 vs 0.298), within noise. This validates that 5-token speculation with verification is a free latency win, not a quality trade.
- The launcher booted clean and held VRAM well under cap.

## What did not work

- TTFT.p50 only improved 1.12× — speculative decoding does not help prefill, which is expected; this scenario isn't TTFT-bound anyway (prefill is just 1024 tokens).

## Suggested follow-ups (not implemented here)

- Bump `num_speculative_tokens` to 7 (PR brief follow-up): if the n-gram acceptance rate is in the 0.5+ range, more draft tokens would push gains further. Acceptance rate isn't surfaced in vLLM v0.11.0 server-side logs by default, but it can be inferred from `gen_throughput / (1 / TPOT)` once spec is on.
- Try `num_speculative_tokens=3` to compare per-step verification overhead at lower spec depth.
- Port n-gram spec dec to **Scenario D** (4K/2K, conc 4): the 2K decode segment is still output-heavy, but the batch-of-4 verification path is much less efficient than batch-1; worth one quick probe.
- Combine spec dec with the Scenario A FlashAttn-tuned launcher (chunked prefill + larger batched tokens) for **Scenario D** to chase both the TTFT and TPOT levers in one launcher.
- A draft-model approach (e.g. a small TinyLlama or Mistral 1B draft) on B could probably exceed n-gram if a clean draft model is available — n-gram acceptance is bounded by literal prompt repetition.
