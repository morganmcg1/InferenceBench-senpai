# SENPAI Research Results — ib-20260524-leasefix-r1

Log of reviewed experiments on this advisor branch. Append a new section every time a PR is reviewed (merged, sent back, or closed). Newest at the top.

---

## 2026-05-24 23:24 — PR #76: Sc B vLLM + n-gram-5 spec decoding (no FP8 KV) — **MERGED, WINNER**

- Branch: `fern/scB-vllm-ngram5-fp8kv` (merged squash)
- Hypothesis: vLLM Sc B launcher with n-gram-5 speculative decoding to amplify per-token decode throughput on the concurrency-1 long-output workload. Originally specified `--kv-cache-dtype fp8`; student self-corrected to drop FP8 KV after diagnosing the FA+FP8KV-on-Blackwell incompatibility (advisor-blessed).
- Result table:

| Metric | Full eval (rnc0c1by, 64 burst) | Quick relaunch (jqilihl9, 4 burst) |
|---|---:|---:|
| **scenario/B/speedup_over_pytorch** | **2.6937x** | 3.5134x |
| 1/tpot.p50 (tok/s) | 107.09 | 139.68 |
| tpot.p50 (ms) | 9.338 | 7.159 |
| ttft.p50 (s) | 0.0641 | 0.0641 |
| generation_throughput (tok/s) | 103.68 | 128.30 |
| success / failure / empty | 64/0/0 | 4/0/0 |
| VRAM peak (MB) | 87,725 | 87,615 |
| Quality MMLU-Pro (n=500) | observed 0.302 / baseline 0.298 / ratio 1.013 / **PASS** | — |

- Conclusion: clear win. The n-gram-5 speculative decoding hypothesis is validated on this scenario/hardware; the corrected (no-FP8-KV) launcher beats vLLM-default's H100 public reference (2.25x) by ~20% even on the slower RTX PRO 6000 shakedown hardware. Clean relaunch reproduced the TPOT magnitude (within small-sample variance), and the quality gate passed convincingly at the authoritative n=500 sample size.
- Notable mechanics observed:
  - vLLM 0.11 V1 engine internally **forces `chunked_prefill_enabled=True` when speculative decoding is on**, regardless of `--no-enable-chunked-prefill`. This is a vLLM hard constraint, not a launcher bug; the TPOT win arrived anyway.
  - `--disable-log-stats` suppresses vLLM's periodic spec-decode stats, so n-gram acceptance rate is not directly logged. Back-of-envelope from 107 tok/s vs ~60 tok/s naïve suggests ~30–50% acceptance. Removing `--disable-log-stats` is a clean follow-up.
  - The 124-timeout pattern observed mid-iteration was the orchestrating Claude shell hitting `SENPAI_TIMEOUT_MINUTES`; the detached `gpu_slot.py` + vLLM server + `evaluate.py` continued and produced the headline result. Important operational lesson: the eval pipeline is independent of the Claude controller.
- Action: squash-merged at 23:24 UTC, `BASELINE.md` updated. Sc B current best: 2.694x.
- Suggested follow-ups (queued for round 2 — do NOT all in this round):
  - n-gram-7 (and n-gram-3) sweep on the same launcher — cheap, direct extension.
  - Capture explicit accept rate by removing `--disable-log-stats`.
  - TRITON_ATTN variant — does not have the FA+FP8KV gating, so FP8 KV becomes viable again as a memory-headroom helper for higher concurrency variants.
  - Medusa or EAGLE draft-model speculative decoding — larger lift toward the SMAC3 15.23x H100 stretch.

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

