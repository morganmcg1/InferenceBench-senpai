# SENPAI Research Results

## 2026-05-24 17:11 — PR #57: Scenario A: vLLM FP8 + FlashInfer attention + no chunked prefill (long-context TTFT)

- **Branch:** `r1-tanjiro/scA-vllm-fp8-flashinfer-prefill`
- **Student:** r1-tanjiro
- **Hypothesis:** Combining vLLM FP8 weights, FP8 KV cache, FlashInfer attention backend, and disabled chunked prefill would give the fastest TTFT for single-stream 8192-token prefill (Scenario A burst c=1). FlashInfer's long-context prefill kernels and FP8 math throughput were expected to push `1/ttft.p50` well above the PyTorch baseline of 2.28 /s.

### Results

| Arm | Backend | KV dtype | Speedup A | Quality | Outcome |
|---|---|---|---:|---|---|
| A (planned) | FLASHINFER | fp8 | — | — | **Crash** — `AssertionError: decode_wrapper._sm_scale == self.scale` at vLLM 0.11 / FlashInfer 0.6.11 on Blackwell SM 120f after JIT compile succeeded |
| B (fallback 1) | FLASH_ATTN | fp8 | — | — | **Crash** — `NotImplementedError: FlashAttention does not support fp8 kv-cache on this device` |
| **C (fallback 2)** | **FLASH_ATTN** | **auto (FP16)** | **1.884x** | **PASS (ratio 1.000)** | **Winner — merged** |

Full Arm C metrics:

| Metric | Value | Baseline |
|---|---:|---:|
| `scenario/A/speedup_over_pytorch` | **1.884x** | 1.000x |
| `ttft.p50` (s) | 0.2328 | 0.4385 |
| `ttft.p90 / p99` (s) | 0.262 / 0.271 | — |
| `tpot.p50` (s) | 0.01792 | 0.02580 |
| `request_throughput` (req/s) | 0.08888 | 0.07089 |
| `generation_throughput` (tok/s) | 53.47 | — |
| MMLU-Pro accuracy | 0.298 | 0.298 |
| Quality ratio | 1.000 | — |
| `failure_count` / `success_count` | 0 / 128 | — |
| `vram_peak_mb` | 92,771 | — |

W&B run: `v051l94c` (group `scA-round1`, project `wandb-applied-ai-team/inferencebench-senpai`)

Launcher path: `senpai/launchers/scA/vllm-fp8-flashinfer-prefill/start_server.sh`

### Analysis and Conclusions

**RTX PRO 6000 (Blackwell SM 120f) confirmed incompatibilities:**
1. **FlashInfer attention runtime failure.** vLLM 0.11 + FlashInfer 0.6.11: the `profile_run` assertion `decode_wrapper._sm_scale == self.scale` fails on Mistral-7B + Blackwell. JIT compilation succeeds (with the NVCC math-library shim the student added), but the runtime phase crashes. Likely a wrapper-reuse or scale-propagation regression in this vLLM/FlashInfer version on SM 120f.
2. **FlashAttention FP8 KV cache rejection.** `NotImplementedError: FlashAttention does not support fp8 kv-cache on this device.` Hard blocked on Blackwell at the vLLM 0.11 level; not a configuration error.

**Positive findings:**
- FP8 weight quantization (`--quantization fp8`) works fine with FLASH_ATTN + FP16 KV.
- Disabling chunked prefill + large max-num-batched-tokens (16384) is the right recipe for burst c=1 TTFT.
- The NVCC math-library shim the student added (`NVCC_PREPEND_FLAGS`) is a useful fix that lands in the launcher and will help other students' FlashInfer JIT attempts.
- 1.884x is a real first-round win — meaningful TTFT improvement — but there is likely more headroom since FP8 KV is unavailable and FlashInfer is blocked.

**Open questions:**
- Would TRITON_ATTN give faster prefill kernels than FLASH_ATTN for 8192-token inputs on Blackwell? (No TRITON_ATTN result yet.)
- Would explicit `--cuda-graph-sizes 1,2,4` reduce decode-step Python overhead for c=1 workload?
- How much of the gap to reference snapshot (~4.5x at H100) is hardware vs software (FP8 KV + FlashInfer unavailable on RTX 6000)?

**Suggested next experiments for Scenario A:**
1. TRITON_ATTN backend + FP16 KV (single-lever swap vs Arm C) → assigned to r1-tanjiro PR #58
2. Explicit cuda-graph-sizes for c=1 (minor tuning on top of merged launcher)
3. `--no-enable-prefix-caching` ablation to confirm caching helps (LongBench prompts may share little prefix)
