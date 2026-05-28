# SENPAI Research Results — `ib-20260528-scen-c-r1`

## 2026-05-28 17:31 — PR #163: SGLang Scenario C boot + tune defaults

- **Branch**: `scen-c-fern/sglang-c-throughput-tune`
- **Hypothesis**: Paper reference shows SGLang default at 51.12x on H100 — highest of all engines for Scenario C. If RTX PRO 6000 SM120 can boot SGLang with `--attention-backend triton` (the only RTX-safe option in the search space) and pass MMLU-Pro quality, even a fraction of the reference 51.12x would be a strong Scenario C baseline. Tuning `max_running_requests`, `mem_fraction_static`, and `schedule-policy` probes additional headroom.

### Setup notes (RTX PRO 6000 SM120 specific)

- SGLang 0.5.9 installed in per-PR venv via `uv venv` + `uv pip install sglang[all]`. `create_engine_venv.py` failed because the stock Python had no `ensurepip`.
- Required apt package: `libnuma1` for `sgl_kernel` to load.
- `--sampling-backend pytorch` required (no FlashInfer on SM120).
- `--attention-backend triton` required (FlashAttention/FlashInfer not used).

### Quick arm results

| Arm | mem_frac | max_running | sched | chunked_prefill | Quick speedup | W&B |
|---|---:|---:|---|---:|---:|---|
| A0 | 0.85 | default | fcfs | 8192 (default) | 3.88x | okdhvh3h |
| A | 0.88 | 256 | fcfs | 4096 | **3.96x** ← promoted | uadpj28k |
| B | 0.88 | 256 | lpm | 4096 | 3.92x | (file wiped) |

LPM vs FCFS scheduling: essentially tied; FCFS marginally ahead. Scenario C requests have no meaningful prefix overlap so RadixAttention/LPM doesn't help.

### Terminal result (Arm A full eval)

| Metric | Value |
|---|---|
| scenario/C/speedup_over_pytorch | **22.18x** |
| geomean req/s | 1.879 req/s |
| quality/mmlu_pro_observed_accuracy | 0.314 (ratio 1.054, PASS) |
| speed requests | 768/768 succeeded |
| VRAM peak | 86.7 GB / 96 GB |
| W&B run | b4xhsfby |
| validation_pass | true |

### Per-profile metrics

| Profile | req/s | gen tok/s | TTFT p50 | TPOT p50 |
|---|---:|---:|---:|---:|
| burst (c=64) | 2.739 | 25.82 | 0.089s | 0.037s |
| poisson (r=32, c=32) | 1.983 | 36.37 | 0.043s | 0.026s |
| constant (r=16, c=16) | 1.221 | 43.90 | 0.039s | 0.022s |

### Analysis and conclusions

- SGLang beats vLLM Arm A (20.84x) by +6.4% on the same RTX PRO 6000 shakedown hardware. Burst-profile req/s 2.74 vs vLLM Arm A's burst is the dominant gap, suggesting SGLang's continuous-batching + RadixAttention layout is better at the wave-of-c=64 pattern even when prefix caching is irrelevant.
- VRAM headroom: 9.3 GB of 96 GB unused. There is room to push `mem_fraction_static` higher (0.90-0.92) and `max_running_requests` up (384-512).
- FCFS ≥ LPM here (Scenario C prompts are diverse).
- Quality is *better* than torch baseline (ratio 1.054), so any speed-favoring tweak should still aim for ratio ≥ 0.95.

### Next directions implied by this result

1. **Push SGLang concurrency**: max_running_requests 384/512, mem_fraction_static 0.90-0.92, chunked-prefill-size 8192 — try to extract the remaining VRAM headroom.
2. **Hardcode the winning config as launcher defaults** so reproduce doesn't depend on caller env vars.
3. **Test SGLang `--enable-torch-compile`** (search space says false, but worth a quick probe).
4. Frieren's PR #165 (Arm D vLLM, chunked-prefill OFF) — still pending; if it wins the steady-state hypothesis, even-money for the lead.

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
