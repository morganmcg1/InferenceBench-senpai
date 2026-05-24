# SENPAI Research Results

## 2026-05-24 07:51 — PR #39: Scenario C: vLLM FP8 weights+KV cache, high-batch throughput sweep

- **Branch:** `r1-frieren/vllm-fp8-batch-scC`
- **Student:** r1-frieren
- **Hypothesis:** Scenario C (high-load, 1K/1K, conc=64 burst) on RTX PRO 6000 Blackwell is bandwidth- and memory-bound at concurrent batch. vLLM with native FP8 weights + FP8 KV cache on Blackwell tensor cores halves both weight-load bandwidth per decode step and per-token KV bandwidth, freeing enough VRAM to push `max-num-seqs` to 256 (above burst concurrency of 64) while chunked prefill folds prefills and decodes in the same scheduler step. FLASH_ATTN used in preference to FlashInfer (safer on Blackwell shakedown).

### Results

| Metric | Value |
|---|---|
| `scenario/C/speedup_over_pytorch` | **46.35x** |
| Geomean req-tput (candidate) | 3.9260 req/s |
| Geomean req-tput (PyTorch baseline) | 0.08471 req/s |
| MMLU-Pro observed accuracy | 0.284 |
| MMLU-Pro baseline accuracy | 0.298 |
| Quality ratio | 0.9530 (PASS, tau=0.95) |
| Total requests | 768/768 (0 failures) |
| VRAM peak | 90,775 MiB (~94 % of 96 GB) |
| Cold relaunch time | ~38 s |
| W&B run | `yiumffcc` |

Per-profile breakdown:

| Profile | Concurrency | req/s | Speedup | TTFT p50 | TPOT p50 | Gen tok/s |
|---|---:|---:|---:|---:|---:|---:|
| burst | 64 | 6.3523 | 75.16x | 33.2 ms | 15.6 ms | 60.3 |
| poisson | 32 | 4.0532 | 47.84x | 25.0 ms | 12.4 ms | 75.7 |
| constant | 16 | 2.3503 | 27.69x | 21.7 ms | 11.0 ms | 86.0 |

### Recipe (`senpai/launchers/C/vllm-fp8-batch/start_server.sh`)

- `--quantization fp8` + `--kv-cache-dtype fp8`
- `--max-num-seqs 256` + `--max-num-batched-tokens 8192`
- `--enable-chunked-prefill`
- `--gpu-memory-utilization 0.92`
- `--block-size 16`
- `VLLM_ATTENTION_BACKEND=FLASH_ATTN`
- vLLM 0.21.0 on Blackwell RTX PRO 6000 (~96 GB)

### Commentary

The 46.35x speedup over PyTorch on Scenario C **nearly matches the public H100 vLLM SMAC3 reference (46.70x)**, which is a strong shakedown result — suggesting RTX PRO 6000 Blackwell performs at H100-parity on throughput-bound serving with FP8. The quality margin is thin (0.9530 vs 0.95 floor). Future runs should consider:

1. **Arm 2a** if quality pressure increases: drop `--quantization fp8`, keep `--kv-cache-dtype fp8`. May widen quality margin at the cost of ~5-10 % throughput.
2. **Higher `--max-num-seqs`** (384-512) to test whether adding more concurrency headroom improves the burst profile further.
3. **SGLang** as an alternative engine — the public SGLang default at 51.12x suggests there is still headroom above 46.35x on this hardware class.

**Verdict: MERGED as first Scenario C baseline (46.35x, quality PASS).**
