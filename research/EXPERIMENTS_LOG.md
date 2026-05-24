# SENPAI Research Results — ib-20260524-hardened-r5

## 2026-05-24 17:45 — PR #63: Scenario A: vLLM + FP8 + FlashInfer prefill

- **Branch**: r5-fern/vllm-fp8-flashinfer-prefill-scA
- **Student**: r5-fern
- **Hypothesis**: Minimize Scenario A TTFT (prefill-bound, 8192 in / 1024 out, burst c=1) via FP8 weights + FP8 KV + FlashInfer prefill kernel on RTX PRO 6000.

| Metric | PyTorch baseline | This run | Relaunch |
|---|---|---|---|
| TTFT.p50 (s) | 0.4385 | **0.2321** | 0.2324 |
| TPOT.p50 (s) | 0.02576 | 0.01789 | 0.01760 |
| Generation throughput (tok/s) | 35.65 | 53.59 | 53.60 |
| **speedup_over_pytorch** | 1.00× | **1.889×** | 1.887× |
| MMLU-Pro accuracy | 0.298 | **0.294** (pass) | 0.298 (pass) |
| VRAM peak (MiB) | — | 92,681 | 92,681 |
| Success rate | 128/128 | 128/128 | 128/128 |

- **W&B runs**: f890wobr (primary), h2ivavpf (relaunch)
- **Launcher**: `senpai/launchers/A/vllm-fp8-flashinfer-prefill-scA/start_server.sh`
- **Final config**: FP8 weights + auto KV (fp16) + FLASH_ATTN backend + chunked-prefill OFF + prefix-caching ON + CUDA graphs ON + max-num-batched-tokens 16384 + max-num-seqs 32 + gpu-mem-util 0.92

**Commentary**: Result is positive (1.89× beats the 1.00× PyTorch-only baseline and the ~1.25× H100 vLLM-default reference), but below the 3–4× target. The planned levers (FlashInfer prefill kernel + FP8 KV) were both blocked by toolchain issues on SM120f:

1. **FP8 KV unreachable**: `FLASH_ATTN` raises `NotImplementedError` for fp8 kv-cache on this device; `FlashInferImpl` asserts on `_sm_scale` under FP8 KV in vLLM 0.11.
2. **FlashInfer unstable**: JIT mixes CUDA 12.8 pip headers (from `runtime_env.sh` CPATH) with CUDA 13.2 nvcc; CPATH must be unset to compile.

The actual speedup comes from: FP8 weight GEMM (halved weight bandwidth) + CUDA graphs + chunked-prefill OFF on a single-stream workload. These are real wins. The gap to ~4× is exactly the two blocked levers — if FP8 KV and a stable FlashInfer are recovered, this launcher is the fallback floor.

**Conclusions**:
- FP8 weights boot cleanly and deliver a measurable speedup.
- FlashInfer + FP8 KV are blocked on SM120f / CUDA 13.2 / vLLM 0.11.
- `--no-enable-chunked-prefill` is the right setting for c=1 workloads.
- `max-num-batched-tokens 16384` ensures the full 8192 prefill fits in one step.
- The next move is to find alternative paths to close the remaining TTFT gap — Triton attention with FP8 KV, or a sweep over larger batched-tokens / speculative-decode for the decode tail.
