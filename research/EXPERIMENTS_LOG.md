# SENPAI Research Results

## 2026-05-24 17:30 — PR #64: Sc D balanced FP8 KV + chunked prefill + TRITON_ATTN

- **Branch:** r2-tanjiro/scenario-d-balanced-fp8kv-chunked
- **Student:** r2-tanjiro
- **Hypothesis:** FP8 KV cache + chunked prefill + FlashInfer attention backend gives a 2× speedup on Scenario D (general balanced: 4K→2K, concurrency=4). Bundled FlashInfer with other vLLM tuning levers.

### Results

| Metric | Value | vs PyTorch |
|---|---:|---:|
| scenario/D/speedup_over_pytorch | **1.284×** | +28.4% |
| Raw geomean (1/ttft, 1/tpot, rps) | 2.478 | PyTorch: 1.930 |
| ttft.p50 (s) | 0.1922 | −9.5% |
| tpot.p50 (s) | 0.0165 | −34.2% |
| tpot.p99 (s) | 0.0456 | −64.1% |
| request_throughput_req_per_s | 0.04828 | +26.3% |
| generation_throughput_tokens_per_s | 57.1 | +49.5% |
| quality/mmlu_pro_observed_accuracy | 0.284 | ratio=0.953 (PASS, tau=0.95) |
| VRAM peak | 90.9 GB | |

- **W&B run:** `wandb-applied-ai-team/inferencebench-senpai/runs/m4w2qrxd`
- **Final launcher:** `senpai/launchers/D/balanced-fp8kv-chunked/start_server.sh`
- **Flags:** `--kv-cache-dtype fp8 --enable-chunked-prefill --max-num-batched-tokens 8192 --max-num-seqs 64 --gpu-memory-utilization 0.92`
- **Attention backend:** TRITON_ATTN (via `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_DISABLE_FLASHINFER_PREFILL=1` set by runtime_env.sh)
- **Status:** MERGED (first Sc D winner)

### Analysis

The original FlashInfer backend prescription had to be dropped due to RTX PRO
6000 Blackwell (sm_120) incompatibility: FlashInfer JIT fails with CCCL header
conflict between CUDA 12.8 Python wheels and on-pod nvcc 13.2. FlashAttention
also rejects fp8 KV dtype on this device. The student adapted using TRITON_ATTN,
which supports fp8 KV on Blackwell.

Key finding: **TRITON_ATTN + FP8 KV cache is the bootable recipe on this pod.**
The FP8 KV cache is doing the heavy lifting — tpot.p50 dropped 34% and gen
throughput climbed 49%, consistent with halving KV bandwidth for decode at
concurrency=4.

The quality margin is thin (ratio=0.953, just 0.003 above the 0.95 tau floor).
Any further FP8 quantization (weight quant) would likely breach the quality
gate. This limits the next-lever options for Sc D.

The 1.284× speedup is modest relative to the H100 SMAC3 reference (5.69×).
The gap reflects both hardware differences (H100 vs RTX PRO 6000 bandwidth) and
the infrastructure constraints (no FlashInfer). Follow-ups should target prefix
caching and n-gram speculative decoding on top of this proven recipe.

### Infrastructure discoveries (affects all future assignments)

1. **FlashInfer JIT broken on sm_120:** `VLLM_ATTENTION_BACKEND=FLASHINFER` causes
   nvcc compilation failure. Root cause: CUDA 12.8 Python wheels in CPATH conflict
   with on-pod nvcc 13.2 CCCL headers. Fixed by patching runtime_env.sh to set
   `VLLM_USE_FLASHINFER_SAMPLER=0` and `VLLM_DISABLE_FLASHINFER_PREFILL=1`.
2. **FlashAttention rejects fp8 KV:** FlashAttention backend (default) does not
   support `--kv-cache-dtype fp8` on this GPU. TRITON_ATTN does.
3. **max_model_len=131072 rejected by vLLM 0.11:** Mistral-7B config is 32768.
   runtime_env.sh now defaults `INFERENCE_BENCH_MAX_MODEL_LEN=32768`.
