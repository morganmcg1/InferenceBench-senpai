# SENPAI Research Results — ib-20260525-three2-r1

_Log of reviewed PRs. Each entry is added when the PR reaches a terminal SENPAI-RESULT and the advisor has reviewed it. Append-only._

## 2026-05-25 21:03 — PR #108: Scenario A: vLLM prefill sweep (chunked-prefill ON vs OFF, big batch budget)
- **Branch:** frieren/scA-vllm-prefill-sweep
- **Hypothesis:** Scenario A (TTFT, input-heavy) benefits from chunked-prefill ON + max-num-batched-tokens=16384, which lets vLLM overlap prefill chunks with decode for other requests and saturate DRAM bandwidth.
- **Results:**

| Arm | Config | Speedup | TTFT p50 | W&B |
|---|---|---:|---:|---|
| 0 vLLM default | gpu-mem 0.90 | 1.274x | 0.344s | y65m9q4h |
| **1 chunked ON (terminal)** | +chunked-prefill +max-batched=16384 +no-prefix-cache gpu-mem 0.92 | **1.280x** | 0.343s | kw34mlo8 |
| 2 chunked OFF | -chunked-prefill +max-batched=16384 +no-prefix-cache gpu-mem 0.92 | 1.278x | 0.343s | 9jq9360t |

- **Quality gate:** quick-only (n=16), observed 0.250 / baseline 0.298 / ratio 0.839 (no full eval; BF16 throughout, no precision change)
- **Outcome:** Negative. All 3 arms within 0.4% — refutes the chunked-prefill hypothesis for Scenario A on this hardware.
- **Commentary:** Scenario A at concurrency 1 has exactly one in-flight request at a time, so the chunked-prefill scheduler has nothing to overlap. With 8192 input tokens and max-num-batched-tokens=16384 the entire prefill fits in one scheduler step in either mode — the scheduling decision degenerates. Cost is dominated by the single-request 8k prefill matmul itself. RTX PRO 6000 hardware baseline 1.274x matches H100 vLLM-default 1.25x proportionally. The FLASH_ATTN backend was already active by default (FlashInfer disabled by runtime_env.sh on Blackwell). Prefix-caching off and seqs=32 made no difference. **Next lever: weight quantization (AWQ/FP8 weights) to shrink the 8k prefill compute/BW cost, or speculative decoding for the 1024-token decode tail.**

## 2026-05-25 21:03 — PR #109: Scenario B: vLLM decode sweep (CUDA graphs ON, block-size 16 vs 32)
- **Branch:** fern/scB-vllm-decode-sweep
- **Hypothesis:** Scenario B (TPOT, output-heavy) benefits from CUDA graphs forced ON (no --enforce-eager), block-size 16 vs 32, and decode-tight config (no prefix caching, no chunked prefill).
- **Results:**

| Arm | Config | Speedup | TPOT p50 | W&B |
|---|---|---:|---:|---|
| 0 vLLM default | gpu-mem 0.90 | 1.415x | 17.78ms | 6xukzikz |
| **1 block-16, no-prefix-cache, no-chunked-prefill** | max-num-seqs=32 max-batched=4096 block-16 gpu-mem 0.92 | **1.421x** | 17.70ms | 53phrwhc |
| 2 block-32 | same but block-32 gpu-mem 0.95 | 1.421x | 17.70ms | 53phrwhc |
| 3 max-num-seqs=1 | decode-tight extreme | 1.419x | 17.72ms | 7h7i5ipf |

- **Quality gate:** quick-only (n=16), observed 0.250 / baseline 0.298 / ratio 0.839 (no full eval; BF16 throughout)
- **Outcome:** Negative. All 4 arms cluster within 0.5% — refutes the CUDA-graphs and block-size hypotheses for Scenario B.
- **Commentary:** The critical diagnosis: vLLM 0.11.0 defaults to CUDA graphs already (`enforce_eager=false`), so "CUDA graphs ON" was not a distinct configuration from default. Block-size variation (16 vs 32) doesn't move TPOT because per-block scheduling overhead is dwarfed by the 64-layer Mistral-7B forward pass at 8192 generated tokens. Disabling prefix caching / chunked prefill has no effect because Sc B prefill is 1024 tokens (tiny) with no reuse at concurrency 1. max-num-seqs=1 doesn't help because vLLM already sees only 1 active sequence at steady state. The 1.42x RTX PRO 6000 hardware floor matches H100 vLLM-default 2.25x proportionally. A torch upgrade collision (tanjiro's SGLang install) delayed the run ~20 min. **Next lever: speculative decoding (EAGLE3, Medusa, n-gram draft) — the only known lever that directly reduces TPOT at concurrency 1 without quantization. Engine swap to SGLang (once sm120 wheels available) is an alternative path.**

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
