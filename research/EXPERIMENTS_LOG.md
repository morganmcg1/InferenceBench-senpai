# SENPAI Research Results — ib-20260524-leasefix-r1

Log of reviewed experiments on this advisor branch. Append a new section every time a PR is reviewed (merged, sent back, or closed). Newest at the top.

---

## 2026-05-24 22:11 — PR #75: Sc A vLLM prefill + FP8 KV (FLASH_ATTN) — CLOSED, negative (boot failure)

- Branch: `frieren/scA-vllm-prefill-fp8kv`
- Hypothesis: vLLM Sc A launcher pairing `VLLM_ATTENTION_BACKEND=FLASH_ATTN` with `--kv-cache-dtype fp8`, plus `--enable-chunked-prefill`, `--max-num-batched-tokens 16384`, `--enable-prefix-caching`, CUDA graphs, `block_size=32`, `max_num_seqs=32`, `gpu_memory_utilization=0.95`.
- Result table:

| Metric | Value | Notes |
|---|---|---|
| Boot | FAILED | `NotImplementedError: FlashAttention does not support fp8 kv-cache on this device.` |
| Scenario A speedup over PyTorch | n/a | engine never served a request |
| Quality MMLU-Pro | n/a | gate not evaluated |
| W&B runs | none | server never came up |
| GPU slot use | 1 quick-eval acquisition, clean release | TTL 600s |
| PyTorch baseline (preflight, seed 248, this hardware) | `ttft.p50=0.4385s`, `ttft.p90=0.5089s`, `ttft.p99=0.9756s` | recorded for the new Sc A PR |

- Conclusion: hard hardware/version incompatibility, not a serving regression. vLLM 0.11's FA+FP8KV path is gated to FA3 + SM 9.0 (Hopper); RTX PRO 6000 is SM 12.0 (Blackwell) where vLLM forces FA back to v2, which does not implement FP8 KV (source pointers in PR #75's terminal comment by frieren).
- Action: closed PR #75. No code merged; the new launcher file was confined to `senpai/launchers/scenario_a/vllm-prefill-fp8kv/start_server.sh` and is left on the closed branch.
- Lessons learned (added to `CURRENT_RESEARCH_STATE.md`):
  - **FLASH_ATTN + FP8 KV is impossible on this hardware.** Do not assign that combo again in this run.
  - **TRITON_ATTN is the canonical Blackwell path for FP8 KV** (`platforms/cuda.py` auto-routes there for non-Hopper); tanjiro (PR #87) is exercising it for Sc C.
  - **FlashInfer + FP8 KV** is technically supported but the runtime helper disables FlashInfer; gated on advisor/human decision.
- Follow-ups:
  - PR #89 (frieren) — corrected Sc A launcher, FLASH_ATTN, drop FP8 KV. Isolates the chunked-prefill + large `max_num_batched_tokens` + CUDA graphs hypothesis.
  - Future: `vllm-prefill-trionattn-fp8kv` for Sc A once tanjiro's TRITON_ATTN behavior on Blackwell prefill is known.

