# SENPAI Research Results — `ib-20260525-three1-r1`

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
