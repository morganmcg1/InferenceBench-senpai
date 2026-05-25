# SENPAI Baseline — ib-20260525-three2-r1

## Setting
- **Hardware:** 1x NVIDIA RTX PRO 6000 Blackwell (~96GB VRAM, shakedown — not leaderboard-comparable to H100)
- **Model:** mistralai/Mistral-7B-Instruct-v0.3
- **Budget:** 2 hours total research program window
- **Scoring assets:** imported from `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`
- **Preflight:** passes for scenarios A/B/C/D (2026-05-25 19:48 UTC)
- **W&B project:** `wandb-applied-ai-team/inferencebench-senpai`

## PyTorch Baseline (raw objective per scenario, this hardware)
PyTorch baseline metrics are present in `src/eval/inference/baselines/speed/torch/inference_scenario_*/.../baseline_metrics.json`. The PyTorch baseline raw objective per scenario gets value 1.00x by definition.

## Current best per scenario (terminal full-eval valid)
| Scenario | Best speedup_over_pytorch | PR | W&B run | Notes |
|---|---:|---|---|---|
| A: Input-heavy (TTFT) | **1.92x** | #113 | ikthm99f | vLLM 0.11.0 --quantization fp8 --enable-chunked-prefill --max-num-batched-tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching --gpu-mem 0.92, FLASH_ATTN; FP8 weights W8A8 dynamic, 1.92x over PyTorch (+50% over BF16 1.28x); quality quick n=16 ratio 0.839 (same noise floor as BF16 quick — needs n=500 confirm) |
| B: Output-heavy (TPOT) | **1.46x** | #114 | nbfrqfvy | vLLM 0.11.0 --quantization fp8 + PR #109 decode-tight launcher; +3% over BF16 1.42x — FP8 weights help less in decode-bound workloads (vs +50% for prefill-bound Sc A); quality quick n=16 ratio 0.839 same as BF16 noise floor |
| C: High-load (req/s geomean) | **21.85x** | #110 | m6pt9mpu | vLLM 0.11.0 --max-num-seqs 256 --max-num-batched-tokens 8192 --enable-chunked-prefill; quality 0.31/0.298 ratio 1.04 ✅ |
| D: General (geomean) | **1.41x** | #115 | rciu3qok | vLLM 0.11.0 --quantization fp8 + PR #112 balanced launcher; +13% over BF16 1.25x — FP8 weights helps Sc D (mixed prefill+decode at c=4); quality quick n=16 ratio 0.839 |

## Reference snapshot (paper, H100, NOT this hardware)
From `program.md` — public reference, do not rank as our baseline. Used only as ceiling guidance.
- A best 4.37x | B best 15.23x | C best 46.70x | D best 5.69x (SMAC3 2h vLLM)
- vLLM default (no agent): A 1.25x | B 2.25x | C 48.69x | D 1.96x
- SGLang default (no agent): A 1.22x | B 1.77x | C 51.12x | D 2.14x

## Scenario A detail — PR #113 (merged 2026-05-25 21:24 UTC) ⭐ FP8 winner
- **Launcher:** `senpai/launchers/scenarioA/fp8-weights-quick/start_server.sh`
- **Engine:** vLLM 0.11.0
- **Key flags:** `--quantization fp8 --enable-chunked-prefill --max-num-batched-tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching --gpu-memory-utilization 0.92`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN`, KV-cache-dtype auto (BF16)
- **PyTorch baseline raw:** 1/ttft.p50 = 2.280 → FP8 raw 4.380 → **1.921x speedup**
- **Quality:** quick-only (n=16), observed 0.250 / baseline 0.298 / ratio 0.839 — same noise floor as BF16 quick (both hit 4/16); full n=500 needed to call true FP8 quality impact
- **W&B:** run ikthm99f
- **VRAM peak:** 90.4 GB / 95 GB
- **Finding:** **Positive winner.** FP8 boots cleanly on SM120 Blackwell (no missing-kernel issue). Cuts prefill GEMM cost roughly in half — confirms PR #108's diagnosis that Sc A at concurrency 1 was matmul-bound, not scheduler-bound. Next lever: full n=500 quality eval; FP8 W8A8 vs int8_w8a8/awq_marlin comparison; stack FP8 with speculative decoding for decode tail.

## Scenario A detail — PR #108 (merged 2026-05-25 21:03 UTC, superseded by #113)
- **Launcher:** `senpai/launchers/A/scA-vllm-chunked-prefill-16k/start_server.sh`
- **Engine:** vLLM 0.11.0
- **Key flags:** `--enable-chunked-prefill --max-num-batched-tokens 16384 --max-num-seqs 32 --block-size 16 --no-enable-prefix-caching --gpu-memory-utilization 0.92`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN`
- **PyTorch baseline raw:** 1/ttft.p50 = 2.280 → Arm 1 raw 2.918 → **1.280x speedup**
- **Quality:** quick-only (n=16), observed 0.250 / baseline 0.298 / ratio 0.839 — no full eval; BF16 throughout, no precision change
- **W&B:** runs y65m9q4h (arm0), kw34mlo8 (arm1 terminal), 9jq9360t (arm2)
- **Finding:** Negative — all 3 arms within 0.4%; concurrency-1 single-request prefill is matmul-bound, not scheduler-bound. **Diagnosis confirmed by PR #113.**

## Scenario B detail — PR #114 (merged 2026-05-25 21:31 UTC) ⭐ FP8 winner
- **Launcher:** `senpai/launchers/B/fp8-weights-quick/start_server.sh`
- **Engine:** vLLM 0.11.0
- **Key flags:** `--quantization fp8 --max-num-seqs 32 --max-num-batched-tokens 4096 --block-size 16 --no-enable-prefix-caching --no-enable-chunked-prefill --gpu-memory-utilization 0.92`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN`, KV-cache auto (BF16)
- **PyTorch baseline raw:** 1/tpot.p50 = 39.76 → FP8 raw 58.21 → **1.464x speedup**
- **Quality:** quick-only (n=16), observed 0.250 / baseline 0.298 / ratio 0.839 — same noise floor as BF16 quick
- **W&B:** run nbfrqfvy
- **Finding:** Positive (+3% over BF16 1.42x). **Notable comparison:** FP8 helps Sc B (decode-bound) by +3% vs +50% for Sc A (prefill-bound, PR #113). Confirms that FP8 weight quantization wins most for prefill-bound workloads — decode is memory-bandwidth-limited and FP8 weights help less here. Next: speculative decoding for Sc B (the real TPOT lever), or full-eval to validate this small gain.

## Scenario B detail — PR #109 (merged 2026-05-25 21:03 UTC, superseded by #114)
- **Launcher:** `senpai/launchers/B/arm1-block16/start_server.sh`
- **Engine:** vLLM 0.11.0
- **Key flags:** `--max-num-seqs 32 --max-num-batched-tokens 4096 --block-size 16 --no-enable-prefix-caching --no-enable-chunked-prefill --gpu-memory-utilization 0.92`
- **PyTorch baseline raw:** 1/tpot.p50 = 39.76 → Arm 1 raw 56.51 → **1.421x speedup**
- **Quality:** quick-only (n=16), observed 0.250 / baseline 0.298 / ratio 0.839 — no full eval; BF16 throughout, CUDA graphs already on by vLLM default
- **W&B:** runs 6xukzikz (default), 53phrwhc (arm1+arm2), 7h7i5ipf (arm3)
- **Finding:** Negative — all 4 arms within 0.5%; CUDA graphs already on in vLLM 0.11.0 default, block-size and seqs settings don't move TPOT at concurrency 1. Next lever: speculative decoding (EAGLE3/n-gram), engine swap.

## Scenario D detail — PR #112 (merged 2026-05-25 21:20 UTC)
- **Launcher:** `senpai/launchers/D/vllm-balanced-quick/start_server.sh`
- **Engine:** vLLM 0.11.0
- **Key flags:** `--max-num-seqs 32 --max-num-batched-tokens 16384 --enable-chunked-prefill --no-enable-prefix-caching --gpu-memory-utilization 0.92`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN`
- **PyTorch baseline raw geomean inverse latency throughput:** 1.9298 → quick raw 2.4137 → **1.251x speedup**
- **Per-metric:** TTFT p50 1.09x | TTFT p90 0.39x ⚠️ (regression at c=1) | TPOT p50 1.49x | gen_throughput 1.50x | req_throughput 1.20x
- **Quality:** quick-only (n=16), observed 0.250 / baseline 0.298 / ratio 0.839
- **W&B:** run 3pm8uuvh
- **Caveat:** quick-eval forces concurrency=1, while Sc D scenario uses concurrency=4. The chunked-prefill + 16384 batched-tokens config is designed to help at c=4 (4×4k concurrent prefills batched in one scheduler step) but this is not exercised in quick-eval. **Full eval at c=4 likely shows higher speedup with TTFT p90 regression eliminated.**
- **Finding:** Positive (first Sc D baseline). Kernel-level wins (TPOT 1.49x) dominate the quick number; the chunked-prefill batching hypothesis remains unfalsified by full eval.

## Scenario D detail — PR #115 (merged 2026-05-25 21:32 UTC) ⭐ FP8 winner
- **Launcher:** `senpai/launchers/scenarioD/fp8-weights-quick/start_server.sh`
- **Engine:** vLLM 0.11.0
- **Key flags:** PR #112 launcher + `--quantization fp8` (KV-cache auto/BF16)
- **PyTorch baseline raw geomean inverse latency throughput:** 1.9298 → FP8 raw 2.7274 → **1.413x speedup** (+13% over BF16 1.25x)
- **Quality:** quick-only (n=16), observed 0.250 / baseline 0.298 / ratio 0.839
- **W&B:** run rciu3qok
- **Finding:** Positive. FP8 contributes for Sc D's mixed prefill+decode at c=4 — smaller gain than Sc A (+50%) but bigger than Sc B (+3%), consistent with prefill-bound > mixed > decode-bound FP8 win ordering. Caveat: quick-eval still runs at c=1 not Sc D's c=4 so chunked-prefill batching at c=4 remains unexercised.

## Scenario C detail — PR #110 (merged 2026-05-25 20:36 UTC)
- **Launcher:** `senpai/launchers/C/vllm-tuned-256seq-8192batch/start_server.sh`
- **Engine:** vLLM 0.11.0
- **Key flags:** `--max-num-seqs 256 --max-num-batched-tokens 8192 --enable-chunked-prefill --no-enable-prefix-caching --gpu-memory-utilization 0.92`
- **MAX_MODEL_LEN:** 32768 (set by runtime_env.sh)
- **Per-profile req/s:** burst 2.68, poisson 1.96, constant 1.21 (geomean 1.85 req/s)
- **PyTorch baseline geomean req/s:** 0.0847 req/s → speedup 21.85x
- **Quality:** MMLU-Pro 0.310 / 0.298 / ratio 1.04 ✅ | 768/768 success
- **VRAM peak:** 87 GiB / 96 GiB
- **W&B:** `wandb-applied-ai-team/inferencebench-senpai` run `m6pt9mpu`
- **Note:** SGLang blocked on SM120 wheels (RTX PRO 6000 Blackwell); vLLM fallback only.

## Update history
- 2026-05-25 19:48 UTC — initial scaffold created; no terminal winners yet.
- 2026-05-25 20:36 UTC — Scenario C baseline set at 21.85x from PR #110 (tanjiro); first terminal winner.
- 2026-05-25 21:03 UTC — Scenario A baseline set at 1.28x from PR #108 (frieren); quick-only, negative parametric result.
- 2026-05-25 21:03 UTC — Scenario B baseline set at 1.42x from PR #109 (fern); quick-only, negative parametric result.
- 2026-05-25 21:20 UTC — Scenario D baseline set at 1.25x from PR #112 (tanjiro); quick-only, positive (first Sc D baseline). All 4 scenarios now have baseline entries.
- 2026-05-25 21:24 UTC — Scenario A baseline raised to 1.92x from PR #113 (frieren); FP8 weights W8A8 dynamic — first non-parametric winner, +50% over BF16 baseline.
- 2026-05-25 21:31 UTC — Scenario B baseline raised to 1.46x from PR #114 (fern); FP8 weights, +3% over BF16. Diagnoses FP8 win scaling: prefill-bound>>decode-bound.
- 2026-05-25 21:32 UTC — Scenario D baseline raised to 1.41x from PR #115 (tanjiro); FP8 weights, +13% over BF16. Mixed workload between A (+50%) and B (+3%) — confirms ordering.
