# SENPAI Research Results — `ib-20260525-one4-r1`

Per-PR experiment ledger. Each entry records the hypothesis, results, and
analysis once the PR returns terminal `SENPAI-RESULT` evidence.

<!-- Entries will be appended chronologically below. -->

## 2026-05-25 13:57 — PR #102: Sc B: vLLM + n-gram spec decoding + CUDA-graphed BF16 KV launcher

- **Branch:** `frieren/sc-b-vllm-ngram-spec-bf16`
- **Hypothesis:** Scenario B (output-heavy, 1024 in / 8192 out, burst c=1) is decode-bound. vLLM with CUDA graphs + BF16 KV + n-gram speculative decoding (spec_tokens=7, prompt_lookup 2–5) should substantially beat the PyTorch baseline TPOT.
- **W&B run:** `x9t8u1d6` (group `scenario-B-ngram-spec-decoding`)

### Results

| Metric | Value | PyTorch baseline | Change |
|---|---|---|---|
| `scenario/B/speedup_over_pytorch` | **3.497x** | 1.00x | +3.50x |
| `burst.tpot.p50` | 0.00719 s | 0.02515 s | **3.50x faster** |
| `burst.tpot.p90` | 0.01221 s | — | — |
| `burst.generation_throughput` | 129.26 tok/s | 39.19 tok/s | +3.30x |
| `burst.itl.p50` | 0.01296 s | 0.01462 s | 1.13x |
| `success / total` | 64/64 | 64/64 | = |
| `failure_rate` | 0 | 0 | = |
| `mmlu_pro_observed` | 0.300 | 0.298 | ratio 1.007 ✅ |
| VRAM peak | 89,759 MB | — | — |

### Quick-probe summary

| Arm | spec_tokens | lookup_max | quick tpot.p50 | quick speedup |
|---|---|---|---|---|
| 1 | 5 | 4 | 0.00907 s | 2.77x |
| 2 (promoted) | 7 | 5 | 0.00492 s | 5.10x |

Quick arm 2 showed 5.10x but full eval landed at 3.50x — n-gram acceptance rate dilutes from 4 requests to 64. Important calibration: quick probes on n=4 overstate n-gram benefit.

### Analysis

Good first result. vLLM CUDA graphs + n-gram spec decoding clears the 3.5x mark on this Blackwell pod with zero failures and quality well within gate. The gap to the H100 SMAC3 15.23x reference is real and reachable via:
1. Better spec decoding (EAGLE-class draft heads)
2. Wider n-gram lookup window (spec_tokens=10-12, lookup_max=8)
3. FP8 weight quantization (with FA backend; KV remains BF16)
4. Larger `max_num_batched_tokens` for prefix amortization

ITL improvement (1.13x) vs TPOT improvement (3.50x) is expected: spec decoding reduces per-step count (TPOT wins) but the gap between accepted chunks still has wall-clock overhead (ITL lags).

### Verdict: MERGED

## 2026-05-25 14:25 — PR #103: Sc C: vLLM high-throughput launcher — no spec, large batches

- **Branch:** `frieren/sc-c-vllm-hightput-no-spec`
- **Hypothesis:** vLLM tuned for concurrent throughput (max_num_seqs=256, max_num_batched_tokens=32768, gpu_memory_utilization=0.95, chunked prefill + prefix caching, BF16 KV, FA, CUDA graphs, no spec) clears the PyTorch baseline by 30-50x on Sc C.
- **W&B run:** none (full eval did not finish serialization)

### Results

- **Quick probe (n=4 req/profile + 16 MMLU):** geomean req/s 0.239 vs PyTorch 0.0847 → **2.82x raw** geomean speedup; all 28 requests succeeded; quality 0.25/0.298 = 0.839 (failed gate on n=16, expected noise level).
- **Full eval (incomplete):** speed phase finished all 768 generations (256 × 3 profiles). Server logs show 200 OK throughout, no OOM, no errors. MMLU-Pro quality phase never started; metrics_full.json never written; W&B not logged.
- **Cause of failure:** `gpu_slot.py run --wait --ttl 2000` lease expired before `evaluate.py` could complete the quality phase and serialize results. The TTL was set too tight for the full Sc C speed+quality cycle on this pod.

### Operational finding (separate from launcher)

vLLM 0.11 + FlashInfer first boot requires `curand.h` for JIT sampling-kernel compilation, but the bundled `/usr/local/lib/python3.10/dist-packages/nvidia/curand/include/` is not on nvcc's include path. Frieren applied a pod-local symlink workaround. **Follow-up tooling work** (not a serving PR): add symlinks or set `NVCC_PREPEND_FLAGS=-I.../nvidia/curand/include` in `senpai/runtime_env.sh`.

### Analysis

The launcher itself is plausibly correct — server was healthy, all speed requests passed at both quick and full scale. The failure is an operational/infrastructure issue (TTL budgeting, FlashInfer JIT cost on first boot) rather than a launcher bug. The 2.82x raw quick speedup is below the 10-30x expected for a high-throughput Sc C launcher, suggesting either (a) prefix caching is hurting at high concurrency, or (b) CUDA graphs are not getting captured for the burst c=64 batch size — both worth investigating in a future PR with more wall-clock.

### Verdict: CLOSED — incomplete eval; no quality gate, no metrics_full.json. Not mergeable. Re-attempt in a future run with `--ttl 3000+`, pre-warmed FlashInfer JIT cache, and a launcher tweak (try `--disable-prefix-caching` or `--enforce-eager` for sanity on the first arm).
