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
