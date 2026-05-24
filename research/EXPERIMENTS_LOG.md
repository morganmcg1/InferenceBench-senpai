# SENPAI Research Results — ib-20260524-hardened-r4

## 2026-05-24 16:52 — PR #61: vLLM prefill launcher for Scenario A (input-heavy)

- **Branch:** `r4-fern/vllm-prefill-a`
- **Student:** r4-fern
- **Hypothesis:** Chunked prefill + large max-num-batched-tokens (16384) + prefix
  caching + FP8 KV cache on RTX PRO 6000 Blackwell reduces TTFT for 8k-token
  input-heavy requests (Scenario A). FlashAttention backend + CUDA graphs.
- **Merged:** yes (squash-merge)

### Results

| Metric | Candidate | PyTorch baseline | Ratio |
|---|---:|---:|---:|
| `scenario/A/speedup_over_pytorch` | **1.261x** | 1.00x | winner |
| TTFT p50 | 0.348 s | 0.4385 s | −20.7% |
| TTFT p90 | 0.397 s | — | — |
| TPOT p50 | 0.0172 s | 0.0258 s | −33.3% |
| req/s | 0.0906 | 0.0709 | +27.8% |
| gen_throughput | 55.28 tok/s | — | — |
| VRAM peak | 90,217 MiB | — | — |
| MMLU-Pro accuracy | 0.296 (ratio 0.9933, tau 0.95) | 0.298 | **quality pass** |
| success | 128/128 | 128/128 | 100% |

**W&B run:** `aucwenw1` | group `ib-20260524-hardened-r4-round1`

### Analysis

The result matches the public H100 vLLM-default reference (1.25x), confirming
the recipe is in "stock-tuned vLLM" territory. The primary plan lever (FP8 KV
cache) was vetoed by the hardware: FlashAttention on RTX PRO 6000 Blackwell
explicitly rejects FP8 KV. The student correctly used the pre-authorized rescue
arm (auto KV dtype). With FP8 KV gone, the remaining levers (chunked prefill,
large batched-tokens, prefix caching, CUDA graphs, FLASH_ATTN) yielded 1.261x.

SMAC3 reference on H100 is 4.37x — large gap. Next Scenario A ideas must target
the prefill kernel itself (SGLang Triton attention, INT4/INT8 weights, or a
higher batched-token ceiling) rather than memory bandwidth optimization.

Note: FP8 KV and FlashInfer are **both broken** on this pod's RTX PRO 6000
hardware. All future launcher recipes must avoid these flags.

### Infra findings (applies to all students)

- `VLLM_ATTENTION_BACKEND=FLASHINFER`: JIT fails (CUDA/toolkit ABI mismatch).
- `--kv-cache-dtype fp8`: FlashAttention rejects it on this device.
- `--max-model-len 131072`: vLLM 0.11 rejects (Mistral config is 32768).
- Patched `senpai/runtime_env.sh` now sets `VLLM_USE_FLASHINFER_SAMPLER=0`,
  `VLLM_DISABLE_FLASHINFER_PREFILL=1`, `INFERENCE_BENCH_MAX_MODEL_LEN=32768`.
- Always source `senpai/runtime_env.sh` before any server launch.
- `PROBLEM_DIR` may be relative `target/`; use `export PROBLEM_DIR="$(pwd)"` from
  repo root before running senpai helpers.
