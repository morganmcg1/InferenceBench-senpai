# SENPAI Research Results — ib-20260525-three2-r1

_Log of reviewed PRs. Each entry is added when the PR reaches a terminal SENPAI-RESULT and the advisor has reviewed it. Append-only._

## 2026-05-25 20:36 — PR #110: Scenario C: SGLang throughput sweep (default vs lpm + max-running-requests 256)
- **Branch:** tanjiro/scC-sglang-throughput
- **Hypothesis:** Scenario C (high-load throughput) benefits from SGLang's lpm scheduling and max-running-requests=256. If SGLang blocked, fall back to vLLM with raised concurrency limits.
- **Results:**

| Metric | Value |
|---|---|
| scenario/C/speedup_over_pytorch (full eval) | 21.85x |
| Quality gate | ✅ pass (ratio 1.04, observed 0.31 vs baseline 0.298) |
| burst req/s speedup | 31.70x |
| poisson req/s speedup | 23.10x |
| constant req/s speedup | 14.25x |
| Total success/fail | 768/0 |
| VRAM peak | ~87 GiB / 96 GiB |
| W&B runs | 8c15z1gs (quick), m6pt9mpu (full) |

- **Commentary:** SGLang was completely blocked on RTX PRO 6000 (SM120) due to missing sgl_kernel sm120 precompiled binaries. vLLM fallback with `--max-num-seqs 256 --max-num-batched-tokens 8192 --enable-chunked-prefill` established the first Scenario C terminal baseline at 21.85x. The H100 reference (vLLM-default 48.69x, SMAC3 best 46.70x) is much higher, partly because the RTX PRO 6000 PyTorch baseline throughput is lower than H100's, and partly because no SGLang or search-tuned launcher ran. burst profile (highest concurrency) sees the biggest speedup (31.7x), confirming the hypothesis that concurrency knobs are the main lever. constant profile (lowest concurrency) shows 14.25x — scheduler overhead matters less, and arrival gaps dominate. Next: max-num-seqs sweep at 128/384/512 and chunked-prefill batch-tokens sweep to push the constant-profile floor up.
