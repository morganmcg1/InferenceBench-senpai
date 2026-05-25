# Scenario A — vLLM Prefill Tune

First measured candidate for Scenario A on `ib-20260525-one3-r1`.

## Levers committed in this launcher (`senpai/launchers/A/vllm-prefill-tune/start_server.sh`)

- Engine: vLLM (`vllm.entrypoints.openai.api_server`).
- Attention backend: FlashAttention (`VLLM_ATTENTION_BACKEND=FLASH_ATTN`); FlashInfer prefill stays disabled via `senpai/runtime_env.sh` defaults.
- `--max-model-len 10240`: KV pool just slightly larger than 8K prefill + 1K decode.
- `--gpu-memory-utilization 0.90`.
- `--enable-chunked-prefill` with `--max-num-batched-tokens 16384`: the 8K prefill still goes in one forward pass; chunked prefill keeps the modern scheduler path so the same launcher can be reused at higher concurrency.
- `--max-num-seqs 16`, `--block-size 16`.
- `--no-enable-prefix-caching` (LongBench prompts have no shared prefix).
- BF16 KV cache; CUDA graphs ON (no `--enforce-eager`).
- `--seed 248`, `--tokenizer-mode auto`, `--trust-remote-code`, `--disable-log-stats`.

## Quick-probe arm comparison (4-request burst, 16-question MMLU-Pro)

| Arm | Variant | Quick TTFT.p50 | Quick speedup vs PT | Quick quality (16q) |
|---|---|---:|---:|---:|
| 1 | chunked prefill + 16384 batched tokens | ~0.345 s | 1.272x | 0.250 (quick is unreliable, see full) |
| 2 | monolithic prefill, otherwise identical | ~0.345 s | 1.271x | 0.250 |

Arms tied to within noise on a 4-request quick probe, so chunked-vs-monolithic does **not** materially move TTFT at concurrency 1 on this hardware. Arm 1 was retained as the committed launcher because the chunked-prefill scheduler is the modern default and stays valid as a starting point for scenarios with concurrent decode.

Arm 3 (tight KV pool, `--gpu-memory-utilization 0.85`) was skipped: with Arms 1–2 tied, a smaller KV pool would only confirm headroom for future arms, not pick a winner inside this PR's time budget.

## Full-eval result (Arm 1)

- W&B run: `x6t65ria` (`wandb-applied-ai-team/inferencebench-senpai`).
- Burst (128 reqs): TTFT.p50 = 0.3527 s (vs PyTorch 0.4385 s) → **speedup 1.243x**.
- TPOT.p50 = 0.01721 s (PyTorch 0.0258 s → 1.50x on decode latency too, a free win from the same launcher).
- ITL.p50 = 0.01168 s.
- gen_throughput = 55.4 tok/s; req_throughput = 0.0908 req/s.
- success/failure = 128/0; empty_output = 0.
- Quality: MMLU-Pro observed 0.298 = baseline 0.298, ratio 1.0, **gate PASS**.
- VRAM peak (metrics_full.json + nvidia-smi after run): 90217 MiB on ~96 GiB.
- Clean relaunch: full eval used the supervised `test_server.sh` wrapper, so the launcher was already cold-booted by `launch_supervised_server.sh` inside a fresh process group.

## Suggested follow-ups (do not implement in this PR)

- FP8 KV cache on a backend that actually accepts it on RTX PRO 6000 (FlashAttn here does not).
- `VLLM_ATTENTION_BACKEND=TRITON_ATTN` ablation against the same launcher to see if a different prefill kernel can break through the 1.25x ceiling that this launcher just barely missed.
- Speculative decoding (e.g. EAGLE / Medusa / draft-model) — the TPOT.p50 was already 1.50x and is the bigger lever for follow-up scenarios B and D.
- Port this same launcher to Scenario D (4K/2K, conc 4) — same prefill knobs but with concurrent decode would actually exercise chunked prefill.
