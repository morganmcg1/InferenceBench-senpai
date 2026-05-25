# SENPAI Research Results — `ib-20260525-three1-r1`

---

## 2026-05-25 17:25 UTC — PR #106: Scenario D: tuned vLLM launcher (chunked prefill ON, max-num-seqs 32) — informational, NOT MERGED

- **Student / branch:** tanjiro / `tanjiro/scenario-d-sglang-lpm` (pivoted to vLLM after SGLang sgl-kernel/sglang version skew blocked LPM hypothesis on this pod)
- **Hypothesis:** Tuned vLLM D recipe (`--max-num-seqs 32`, `--enable-chunked-prefill`, `--max-num-batched-tokens 8192`, `--enable-prefix-caching`, `--max-model-len 8192`, `--gpu-memory-utilization 0.95`) separates from default vLLM under Sc. D's c=4 burst concurrency.
- **Result table:**

| Arm | Mode | Speedup | TTFT.p50 | TPOT.p50 | gen tok/s | Quality | W&B run |
|-----|------|--------:|---------:|---------:|----------:|---------|---------|
| fallback default-vLLM-D | quick (4 req, c=1 burst) | 1.241x | 0.1955s | 0.01698s | 56.88 | 0.839 ratio (quick-mode noise, n=16) | `1tryfxbl` |
| tuned-vLLM-D | quick (4 req, c=1 burst) | **1.246x** | 0.1952s | 0.01683s | 58.92 | 0.839 ratio (quick-mode noise, n=16) | `73o7oq1b` |

- **Commentary:** Hypothesis directionally confirmed but not validated — tuned recipe is +0.4% over default at quick scale, well inside noise. The quick eval runs only 4 sequential requests at c=1 burst (despite the "burst" profile name), so scheduling-pressure-sensitive levers (`--max-num-seqs 32`, chunked prefill interleaving) cannot demonstrate their value. Full eval at c=4 burst would be the real test but wall-clock did not permit it this round. SGLang LPM hypothesis remained untested due to documented `sgl-kernel < 0.3.20` / `sglang 0.5.12.post1` version mismatch — operational issue for next launch.
- **Disposition:** **NOT MERGED** — quick eval results are not BASELINE-eligible per the merge contract (full `evaluate.py` + n=500 MMLU-Pro required). Closed as informational. Two launchers banked as round-2 candidates: `senpai/launchers/D/tanjiro-tuned-vllm-balanced/start_server.sh` (commit `efdc787`) for full-eval validation, and `senpai/launchers/D/tanjiro-fallback-vllm-default-d/start_server.sh` (commit `6dca712`) as confirmed default-vLLM-D floor. Student's suggested round-2 follow-up of pairing the tuned-D recipe with frieren's ngram speculative config is the highest-priority candidate.

---

## 2026-05-25 17:03 UTC — PR #105: Scenario A: tuned vLLM prefill launcher (big batched tokens, no chunked prefill)

- **Branch:** `fern/scenario-a-tuned-vllm-prefill`
- **Hypothesis:** Large `--max-num-batched-tokens 16384` collapses TTFT latency for input-heavy (4096-token) prompts under burst c=1; disabling chunked prefill and prefix caching eliminates scheduling overhead.
- **Result table:**

| Arm | Mode | Speedup | TTFT.p50 | TPOT.p50 | gen tok/s | Quality | W&B run |
|-----|------|--------:|---------:|---------:|----------:|---------|---------|
| arm 1 (prefix cache ON)  | quick | 1.271x | — | — | — | 0.25 (noisy) | `sm95i0u2` |
| arm 2 (prefix cache OFF) | quick | 1.280x | — | — | — | 0.25 (noisy) | `osss9jxl` |
| arm 2 (prefix cache OFF) | **full** (128 req + 500 MMLU-Pro) | **1.240x** | **0.3537s** | 0.0172s | 54.74 tok/s | **PASS 1.020 (0.304 vs 0.298)** | **`insngzmf`** |

- **Commentary:** Hypothesis partially confirmed — the tuned launcher improves Sc. A TTFT by 1.24x over PyTorch (baseline ttft.p50 = 0.4385s). The gap to the public H100 reference (1.25x default vLLM, 4.48x tuned ceiling) suggests the primary remaining lever is hardware-level: Blackwell tensor cores can take FP8 weight quant, and the prefill cost on RTX PRO 6000 may respond differently than on H100. The 3-point difference between quick and full (1.28x → 1.24x) is expected sample-variance. Quality is clean. Arm 1 vs arm 2 showed no measurable prefix-caching advantage — likely because burst c=1 with independent requests gets zero prefix cache hits. FP8 arm 3 launcher was staged at `senpai/launchers/A/fern-tuned-vllm-prefill-fp8/start_server.sh` and merged with the branch; deferred to next launch due to wall-clock constraint.
- **Merged:** Yes — new Sc. A baseline for this launch (no incumbent). First measured Sc. A result on RTX PRO 6000 Blackwell.

---
