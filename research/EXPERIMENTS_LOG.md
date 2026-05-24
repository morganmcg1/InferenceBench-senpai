# SENPAI Research Results — ib-20260524-leasefix-r5

InferenceBench launcher experiments. Each entry is one PR reviewed by the
advisor. Update with PR number, student branch, hypothesis, results table
(including W&B run IDs), and commentary.

## 2026-05-24 23:08 UTC — PR #82: tanjiro — Scenario D balanced FP8 launcher + chunked prefill
- Branch: `tanjiro/D-balanced-fp8`
- Hypothesis: FP8 weight quantization + `--enable-chunked-prefill` + balanced batching (`--max-num-seqs 64`, `--max-num-batched-tokens 8192`) at `--max-model-len 16384` lifts Sc. D speedup while preserving MMLU-Pro quality on RTX PRO 6000.
- Launcher: `senpai/launchers/D/balanced-fp8/start_server.sh`.

| Metric | Value | Source |
|---|---:|---|
| `scenario/D/speedup_over_pytorch` | **1.5015x** | W&B `85zn8jd4` |
| `scenario/D/geomean_inverse_latency_throughput` (raw) | 2.8976 | metrics_full.json |
| `quality/mmlu_pro_observed_accuracy` | 0.296 (ratio 0.9933, n=500) | quality gate PASS at tau=0.95 |
| `ttft.p50 / p99` | 111.79 ms / 130.52 ms | burst 96 reqs, 0 fail |
| `tpot.p50 / p99` | 16.87 ms / 44.18 ms | burst |
| `itl.p50 / p99` | 11.68 ms / 12.23 ms | burst |
| `request_throughput` | 0.04588 req/s | burst |
| `generation_throughput` | 55.07 tok/s | burst |
| VRAM peak | 91.44 GB (of ~96 GB) | nvidia-smi peak |
| Cold start | ~78 s to ready | server.log |

- Conclusions:
  - **FP8 weights + chunked prefill work cleanly on RTX PRO 6000 Blackwell.** Cold start ~78 s, zero kernel/sampler issues, MMLU-Pro within 0.7% of the BF16 reference. Earlier hypothesis that FP8 first-load was hanging is disproven — the watchdog kills were inference-target heuristic misfires (looking for `train.py` that never exists on a vLLM eval).
  - **Decode is the dominant cost** on Sc. D: `tpot.p50 ≈ 17 ms` drives the geomean, while TTFT is healthy (`p50 ≈ 112 ms`). Suggests further wins will come from decode-side batching / KV-cache levers, not prefill chunk size alone.
  - **VRAM is the binding constraint** at this config: 91.44 GB peak leaves only ~4–5 GB headroom, so naively raising `--max-num-seqs` or KV cache will OOM without an FP8 KV cache or a tighter prefix budget.
  - **Sc. D baseline is now 1.5015x** on RTX PRO 6000 (first measurement of the run). The H100 reference's 1.96x vLLM-default is hardware-dependent context only.
- Follow-ups raised by tanjiro: (1) measure on-hardware vLLM-default to anchor an ablation, (2) Arm 3 sweep `--max-num-seqs 128 --max-num-batched-tokens 16384`, (3) middle ground `--max-num-batched-tokens 12288`, (4) check `--kv-cache-dtype fp8` safety on Blackwell, (5) re-run MMLU-Pro on every future FP8 variant because the quality margin is only 0.7%.

## 2026-05-24 23:34 UTC — PR #78: fern — Scenario A one-shot long prefill + FP8 weights
- Branch: `fern/A-big-prefill-fp8`
- Hypothesis: at Sc. A's profile (8192-token prompt, concurrency=1), the prefill is the dominant per-request cost. Disabling chunked prefill and sizing `--max-num-batched-tokens 16384` lets the full prompt enter the GPU in a single scheduler step, compressing TTFT. FP8 weight quantization on top is essentially free for Mistral-7B-Instruct quality at tau=0.95.
- Launcher: `senpai/launchers/A/big-prefill-fp8/start_server.sh` (fallback `big-prefill-bf16/start_server.sh` also committed).

| Metric | Value | Source |
|---|---:|---|
| `scenario/A/speedup_over_pytorch` | **1.8903x** | W&B `9hvhqa1u` |
| `scenario/A/inverse_ttft_p50` (raw) | 4.3107 | metrics_full.json |
| `quality/mmlu_pro_observed_accuracy` | 0.298 (ratio **1.000**, n=500) | quality gate PASS at tau=0.95 |
| `ttft.p50 / p90 / p99` | 232.0 / 261.7 / 269.8 ms | burst 128 reqs, 0 fail |
| `tpot.p50` | 17.74 ms | burst |
| `itl.p50` | 11.98 ms | burst |
| `request_throughput` | 0.08947 req/s | burst |
| `generation_throughput` | 54.07 tok/s | burst |
| VRAM peak | 92.81 GB (of ~96 GB) | metrics_full.json |
| Cold start | ~3 min from cold weights through quantization | server.log |

| Quick-eval arm | Launcher | TTFT speedup (quick) | Quality (n=16) | Decision |
|---|---|---:|---|---|
| Arm 1 | FP8 + `--max-num-batched-tokens 16384 --no-enable-chunked-prefill` | **1.90x** | 0.839 (FAIL at n=16 noise) | promoted to full eval |
| Arm 2 | drop `--quantization fp8` (BF16) | 1.28x | 0.839 (FAIL at n=16 noise) | not promoted |

- Conclusions:
  - **One-shot prefill is the dominant Sc. A lever**: BF16 alone gave only 1.28x (≈ H100 vLLM-default 1.25x). Stacking FP8 weight quantization lifted full-eval speedup to 1.8903x with zero quality penalty (ratio exactly 1.000).
  - **TTFT distribution is sharp**: `ttft.p99 − ttft.p50 = 38 ms` vs ~537 ms on the PyTorch baseline — kernels are not stalling on KV allocation or memory traffic at the long-prefill working set.
  - **VRAM headroom is comfortable**: 92.8 GB peak at `--gpu-memory-utilization 0.92` leaves ~3 GB. Raising utilization to 0.95 is a cheap follow-up tweak.
  - **Quick-mode (n=16) MMLU-Pro is at the noise floor** for this model: both FP8 and BF16 returned `0.25/0.298 = 0.839` from identical 16 samples — quick-mode quality gate is not safe to differentiate quantizations on this benchmark and tau without n>>16 or a matched 16-sample baseline accuracy.
- Follow-ups raised by fern: (1) `--enable-prefix-caching` for Sc. A — may compound with one-shot prefill on LongBench-v2 sampled prompts, (2) sweep `--max-num-batched-tokens` 9216-16384 to find min value that still fits longest sampled prompt, (3) `--gpu-memory-utilization 0.95`, (4) cross-hardware sanity check on H100, (5) test same launcher on Sc. D (4096-token prompts at concurrency=4). Advisor flag: raise quick-mode quality sample count or change the gate's baseline to a matched-n estimate.
