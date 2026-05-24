# SENPAI Research State — ib-20260524-leasefix-r1

- **As of:** 2026-05-24 23:31 UTC (round-1 mid-flight; PR #76 merged, two PRs still in flight)
- **Most recent human research direction:** No GitHub Issues open. Programme contract remains `target/program.md`: maximise per-scenario `speedup_over_pytorch` on Mistral-7B-Instruct-v0.3 with a 2-hour wall-clock budget per scenario while passing the MMLU-Pro quality gate.

## Current focus

First measured SENPAI baseline on this branch/hardware has landed: **Sc B = 2.694x** via PR #76. The remaining round-1 priority is to land at least one terminal result on Sc A and Sc C before the 24:00 UTC endpoint, then turn to round-2 follow-ups (sweep n-gram-7 on Sc B; cover Sc D — already kicked off via PR #95 to fern).

We have **1 GPU shared by 3 students** for this round, with the three assignments spread across distinct scenarios to minimise serialisation of full evaluations.

## Round 1 assignments + outcomes

| Student | Scenario | Hypothesis | PR | State |
|---------|----------|------------|----|-------|
| frieren | A — input-heavy, 1/ttft.p50 | vLLM with chunked prefill, `max_num_batched_tokens=16384`, FLASH_ATTN, **FP8 KV cache**, CUDA graphs, prefix caching | #75 | **CLOSED, negative (boot failure)** — FLASH_ATTN+FP8KV impossible on Blackwell SM 12.0 |
| frieren | A — input-heavy, 1/ttft.p50 | corrected: drop FP8 KV, keep FLASH_ATTN + chunked prefill + 16384 max_num_batched_tokens + prefix caching + CUDA graphs | **#89** | WIP; launcher pushed `d56cb60` at 22:37; no measured result, almost certainly slot-blocked behind fern/tanjiro all round. Budget effectively exhausted (~29 min left). |
| fern    | B — output-heavy, 1/tpot.p50 | vLLM with n-gram-5 speculative decoding, CUDA graphs, FLASH_ATTN; **student self-corrected to drop `--kv-cache-dtype fp8`** | **#76** | **MERGED 23:24** — 2.6937x speedup over PyTorch (full eval, 64 burst). Quality MMLU-Pro 0.302 vs 0.298 baseline, ratio 1.013 PASS at n=500. Clean relaunch reproduced TPOT magnitude. Now the live Sc B baseline. |
| fern    | D — balanced burst 4, geomean | round-2 candidate brought forward: vLLM with **n-gram-3 spec** + chunked prefill + prefix caching, `max_num_batched_tokens=8192`, `max_num_seqs=64`. Mid-strength speculative decoding sized for D's burst=4 + 2K output sweet spot. | **#95** | WIP; created 23:28. fern reassigned after Sc B win; will at best commit a launcher this round given ~29 min remaining. Full eval is round-2 work. |
| tanjiro | C — high-load, geomean req/s | vLLM with `max_num_seqs=256`, `max_num_batched_tokens=16384`, chunked prefill, prefix caching, **FP8 KV via TRITON_ATTN** (canonical Blackwell path), CUDA graphs | **#87** | WIP; launcher committed `21b5a2e`. Posted **partial** result at 23:21: scenario/C/speedup_over_pytorch ≈ **24.22x** on first measured pass, MMLU-Pro quality PASS. Needs clean relaunch + terminal `SENPAI-RESULT` before merge. ~29 min budget remaining. |

### Round 1 narrative

- **fern (Sc B → Sc D)**: silent 54-min stall on iter 4 caused by `SENPAI_TIMEOUT_MINUTES` killing the orchestrating Claude shell (`exit 124 after 3284s`); the **detached `gpu_slot.py` + vLLM server + `evaluate.py` continued** and produced the full 64-burst + 500-sample MMLU-Pro result anyway. Operational lesson: the eval pipeline is independent of the Claude controller. Once fern committed the launcher (`1e1718d`), posted the terminal result, and the clean relaunch reproduced TPOT magnitude, PR #76 was squash-merged at 23:24 UTC and `BASELINE.md` updated. fern reassigned to Sc D as PR #95.
- **tanjiro (Sc C)**: same silent-stall pattern, recovered to a partial measured result of ~24.22x speedup with quality PASS. Needs clean relaunch + terminal SENPAI-RESULT to merge.
- **frieren (Sc A)**: launcher landed early (`d56cb60` at 22:37) but never won a slot through fern's and tanjiro's long sessions. No measured result expected in round 1.

### Critical lesson learned in round 1

`VLLM_ATTENTION_BACKEND=FLASH_ATTN` combined with `--kv-cache-dtype fp8` is **impossible on RTX PRO 6000 Blackwell (SM 12.0)** in vLLM 0.11. The FA+FP8KV path is hard-gated to FA3 + Hopper SM 9.0. On Blackwell, vLLM forces FA back to v2, which does not implement FP8 KV. Hard `NotImplementedError` at engine init. Do not assign this combo again in this run.

Canonical Blackwell FP8 KV paths:
1. **TRITON_ATTN backend** — `platforms/cuda.py` auto-routes FP8 KV to TRITON_ATTN on non-Hopper. tanjiro is exercising this on Sc C (PR #87, partial 24.22x).
2. **FlashInfer backend** — supports FP8 KV but `senpai/runtime_env.sh` disables FlashInfer by default on Blackwell shakedown. Re-enabling is allowed if a PR is explicitly testing it.
3. **Drop FP8 KV entirely** — bf16 KV cache is fine on Mistral-7B with 96GB and burst concurrency ≤ 64. fern's winning PR #76 (Sc B) and the corrected frieren PR #89 (Sc A) use this path.

### Other lessons from round 1

- **Detached eval pipelines survive Claude shell SIGTERM**: `gpu_slot.py`, vLLM, and `evaluate.py` continue after the orchestrating Claude session is killed at the 55-min `SENPAI_TIMEOUT_MINUTES` boundary. Students must commit the launcher early so the artifact is durable; the metrics will still appear in W&B even if the controlling shell dies.
- **vLLM 0.11 V1 engine forces `chunked_prefill_enabled=True` when speculative decoding is on**, regardless of `--no-enable-chunked-prefill`. Hard constraint, not a launcher bug.
- **`--disable-log-stats` suppresses spec-decode acceptance rate** in vLLM logs. Removing it is a clean follow-up for any spec-decoding sweep.

Common ground (still): every launcher sources `senpai/runtime_env.sh`, wraps heavy GPU work in `senpai/gpu_slot.py run --wait`, runs quick eval before full, runs a clean relaunch before declaring terminal, and posts a `SENPAI-RESULT` marker per `program.md`.

## Why these three configurations

- **Scenario A (frieren):** TTFT is prefill-dominated. The vLLM 0.11 prefill path benefits most from a batched-token budget that fits the 8K prompt in one or two chunks, plus FP8 KV to clear memory pressure that throttles graph capture. Reference public H100 default vLLM hits only 1.25×; the search-baselines push it to 4.5×. Any clean ≥ 1.5× on RTX PRO 6000 is a first step.
- **Scenario B (fern):** TPOT on a concurrency-1 long-output workload is the cleanest place to play with speculative decoding. n-gram is the safest spec method (exact verification preserves quality), and the SMAC3 H100 result of 15× is largely a speculative-decoding effect. Per-token decode is the metric; CUDA graphs are non-negotiable.
- **Scenario C (tanjiro):** Throughput is batching-dominated. Mistral-7B + 96GB VRAM + FP8 KV cache lets us comfortably push `max_num_seqs` to 256, matching the burst concurrency cap. Public reference is the one scenario where the *default* vLLM is already very strong; modest improvements compound in the aggregate geomean.

## Next research directions (round 2 queue)

Ranked by expected impact relative to the current `BASELINE.md` (Sc B = 2.694x; A/C/D pending):

1. **n-gram-7 sweep on Sc B** — direct follow-up to the merged 2.694x n-gram-5 winner. The H100 SMAC3 reference hits 15.23x on Sc B, almost entirely from speculative decoding; we have ~5.6× of headroom against that. Group runs with `--wandb_group`. Pair with n-gram-3 to confirm the local optimum.
2. **Sc A**: assign once round-1 closes. **frieren PR #89** (FLASH_ATTN bf16 + chunked-prefill + 16384 max_num_batched_tokens + prefix caching + CUDA graphs) is already committed and slot-blocked — extend its budget into round 2 to land the first Sc A number, then iterate. Round-2 follow-up candidate: **FlashInfer backend on Sc A** (`VLLM_ATTENTION_BACKEND=FLASHINFER`), expected 1.3–1.8x TTFT over FA2; first verify the installed FlashInfer's Blackwell sm_100 support.
3. **Sc C terminal**: if tanjiro's clean relaunch doesn't terminate in round 1, finish it in round 2. The partial 24.22x is far above the H100 vLLM-default reference (~1× of the 48.69 req/s baseline) and the H100 SMAC3 stretch is only 46.7 req/s — strong indicator the FP8-KV-via-TRITON_ATTN path is the right canonical config.
4. **Sc D** — fern's PR #95 (n-gram-3 spec + chunked prefill + prefix cache, `max_num_seqs=64`) — full eval to be carried into round 2.
5. **n-gram acceptance-rate visibility** — drop `--disable-log-stats` on the next Sc B variant so we can see acceptance directly instead of inferring from end-to-end TPOT.
6. **Engine bake-off on Sc C** — SGLang RadixAttention vs vLLM prefix caching once tanjiro's vLLM baseline lands; SGLang's H100 default beats vLLM's on throughput-style workloads.
7. **FP8 weights on Sc C** — `--quantization fp8` if FP8 KV holds the quality gate. Biggest VRAM lever, biggest quality risk. Verify `--quantization fp8` CLI compatibility with the installed vLLM build first.
8. **Draft-model speculative decoding on Sc B** — EAGLE / Medusa / draft-Mistral once the n-gram local optimum is mapped; the SMAC3 H100 15.23x almost certainly involves draft models.
9. **Scheduler policy** — vLLM v1 preemption / chunked-prefill admission tweaks; SGLang `--schedule-policy lpm` vs default `fcfs`.
10. **Cross-scenario "universal" launcher** — once we have winners per scenario, run an A–D confirmation on the best general-purpose config and report the aggregate geomean.

## Plateau watch

We are nominally at round 1, so there is no plateau yet. After three rounds of <5% improvement on the same scenario, escalate per the Plateau Protocol in `CLAUDE.md` (change strategy tier; investigate the worst-performing profile; consider non-vLLM engines or kernel-level work).

## Round 2 candidate hypotheses (from researcher-agent)

`research/RESEARCH_IDEAS_2026-05-24_advisor.md` ranks 8 hypotheses. Status after round 1:

1. **n-gram-7 speculative decoding on Sc B** — researcher recommendation now actionable since fern's n-gram-5 PR #76 merged at 2.694x. Top of the round-2 queue.
2. **FlashInfer attention backend on Sc A** — `VLLM_ATTENTION_BACKEND=FLASHINFER` + `VLLM_USE_FLASHINFER_SAMPLER=1` + `VLLM_DISABLE_FLASHINFER_PREFILL=0`. Expected 1.3–1.8× TTFT over the FLASH_ATTN baseline. Risk: Blackwell sm_100 PTX support depends on the installed FlashInfer version; must smoke-test boot first. Schedule **after** frieren's PR #89 (FA bf16) provides the Sc A reference number.
3. **SGLang FP8 weights + RadixAttention on Sc C** — `--quantization fp8 --schedule-policy lpm`. FP8 weights + RadixAttention give ~7 GB free for KV cache. Risk: quality gate with per-tensor FP8 scaling on Mistral GQA; CLI compatibility of `--quantization fp8` on the installed SGLang build. Schedule **after** tanjiro's PR #87 reaches terminal.
4. **SGLang FlashInfer + CUDA graphs on Sc D** — compound winner once round-1 baselines exist. Sequenced **after** fern's PR #95 (vLLM Sc D) returns a baseline.

Open uncertainties flagged by the researcher (still open):
- Installed FlashInfer version's Blackwell sm_100 support — to verify before any FlashInfer launcher in round 2.
- SGLang `--quantization fp8` CLI accepted by the installed build.
- n-gram acceptance rate on LongBench-v2 prompts (Sc B); drop `--disable-log-stats` on the next variant to measure directly.

## Open external context

- Reference numbers in `program.md` are H100; current results are RTX PRO 6000 shakedown only and **not** leaderboard-comparable.
