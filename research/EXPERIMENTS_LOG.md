# SENPAI Research Results — `ib-20260528-scen-c-r1`

## 2026-05-28 17:19 — PR #161: vLLM Scenario C high-concurrency throughput sweep

- **Branch**: `scen-c-frieren/vllm-c-throughput-sweep`
- **Hypothesis**: Pushing vLLM's `max_num_seqs`, `max_num_batched_tokens`, and `gpu_memory_utilization` above the conservative starting-point defaults improves Scenario C geomean throughput. RTX PRO 6000 has ~96GB VRAM vs H100's 80GB, giving headroom. Chunked prefill on + BF16 KV (FP8 KV incompatible with FlashAttention on this hardware).

### Quick arm results

| Arm | max_num_seqs | max_num_batched_tokens | gpu_mem_util | chunked_prefill | Quick geomean speedup | W&B |
|---|---:|---:|---:|---|---:|---|
| A | 384 | 16384 | 0.92 | ON | 2.79x (screening) | j5er5g2l |
| B | 512 | 16384 | 0.93 | ON | 2.76x (screening) | 0szs1hnx |
| D | 384 | 16384 | 0.92 | OFF | 3.80x (screening) | 8ipbzguo |

### Terminal result (Arm A full eval)

| Metric | Value |
|---|---|
| scenario/C/speedup_over_pytorch | **20.84x** |
| geomean req/s | 1.765 req/s |
| PyTorch baseline geomean req/s | 0.0847 req/s |
| quality/mmlu_pro_observed_accuracy | 0.306 (ratio 1.027, PASS) |
| speed requests | 768/768 succeeded |
| W&B run | btpqa2rl |
| validation_pass | true |
| baseline_update_allowed | true |

### Analysis and conclusions

- **Arm A (chunked prefill ON) chosen for full eval** because the quick-mode Arm D advantage (+37% vs Arm A quick) is a small-N artifact: all burst requests arrive concurrently in quick mode (4 req), so chunked prefill just serializes what would otherwise be one batch. Under full eval (256 req/burst, c=64 = 4 waves), chunked prefill should help by overlapping wave-2 prefill with wave-1 decode.
- **Arm B ≈ Arm A** in quick mode — extra max_num_seqs headroom isn't exercised at 4 req/profile.
- VRAM peak stable at 89-91% with gpu_mem_util=0.92.
- **Open question**: Does Arm D (chunked-prefill OFF) beat Arm A in full eval? Needs a full eval run to confirm. If the steady-state argument is right, Arm A should win; if Arm D wins, it implies prefill scheduling is not the bottleneck for 4-wave burst.

### Next directions implied by this result

1. Full eval of Arm D (chunked-prefill OFF) to resolve the open question — high priority.
2. N-gram speculative decoding for the long-decode phase of Scenario C (output_len=1024, ignore_eos=true).
3. SGLang default (PR #163 / fern) booted and passed quality at A0; its full eval is now the most important pending measurement.
