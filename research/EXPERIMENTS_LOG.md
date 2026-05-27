# SENPAI Research Results — ib-20260527-latest3-r1

## 2026-05-27 15:15 — PR #122: vLLM decode-tuned for Scenario B (frieren)

- branch: `frieren/vllm-b-decode-tuned`
- hypothesis: vLLM tuned for decode (CUDA graphs + prefix caching + small max-num-seqs at c=1) beats PyTorch baseline on `1/tpot.p50` by ≥5x on Scenario B
- partial / quick-eval results only — no full eval yet (compete for shared GPU vs tanjiro's full C)

| Arm | Launcher | 1/tpot.p50 raw | Speedup over PyTorch | W&B |
|---|---|---:|---:|---|
| 1 vllm-cudagraph-prefix | `senpai/launchers/B/vllm-cudagraph-prefix/` | 57.27 | 1.44x | `9w5uw7q3` |
| 2 vllm-cudagraph-block32 | `senpai/launchers/B/vllm-cudagraph-block32/` | 56.55 | 1.42x | `3sl0py0b` |
| 3 vllm-ngram-spec | `senpai/launchers/B/vllm-ngram-spec/` | **136.90** | **3.44x** | `08pnkhgg` |

- ngram speculative decoding (num_spec=5, prompt_lookup_min=2, max=4) was the only meaningful lever at c=1 on RTX PRO 6000. Arm 1 confirmed CUDA graphs are on by default in the `vllm_running` starting point, so prefix-caching + max-num-seqs=8 gave only ~1.44x.
- Quick mode = 4 burst requests; full B would take ~65 min at the Arm 3 effective gen rate (~125 tok/s). Tanjiro's full C is using the GPU first, so full B likely will not fit the remaining run window.
- Suggested follow-up: confirm Arm 3 in a follow-up launch with a clean full eval; try `num_speculative_tokens=7` to see if more draft tokens help.

## 2026-05-27 15:30 — PR #124: SGLang throughput-tuned for Scenario C (tanjiro) — MERGED ✓

- branch: `tanjiro/sglang-c-throughput` (merged → ib-20260527-latest3-r1)
- hypothesis: SGLang with mem-fraction-static 0.85 and max-running-requests 128 beats vLLM defaults on Scenario C
- **TERMINAL RESULT — 22.48x speedup over PyTorch baseline, QUALITY PASS**

| Arm | Speedup (full eval) | W&B |
|---|---:|---|
| 2 sglang-mem085-mrr128 (terminal) | **22.48x** | `v478wci3` |

- Per-profile: burst 2.780 req/s, poisson 2.004 req/s, constant 1.239 req/s, geomean=1.9041
- MMLU-Pro: 0.314 vs 0.298, ratio=1.054, n=500. PASS.
- VRAM: 84,285 MB. validate_result.py: validation_pass=true, baseline_update_allowed=true.
- Insight: quick mode (4 req/profile) dramatically under-samples high-concurrency profiles — 3.97x quick vs 22.48x full. Quick mode for C is only useful for launch-failure screening, not speedup estimation.
- Launcher: `senpai/launchers/C/sglang-mem085-mrr128/start_server.sh` (merged). SGLang 0.5.9 with triton attention backend. Note: requires `libnuma1 libnuma-dev` apt packages for `sgl_kernel` import on Ubuntu 22.04 + SM120 GPU.

## 2026-05-27 15:15 — PR #124: SGLang throughput-tuned for Scenario C (tanjiro)

- branch: `tanjiro/sglang-c-throughput`
- hypothesis: SGLang with tuned mem-fraction-static and max-running-requests will match or beat vLLM defaults on high-concurrency throughput (Scenario C)
- partial / quick-eval results only — full eval starting now

| Arm | Launcher | req/s geomean (quick) | Speedup over PyTorch | W&B |
|---|---|---:|---:|---|
| 1 sglang-default | `senpai/launchers/C/sglang-default/` | 0.3365 | 3.97x | `csj0gqi4` |
| 2 sglang-mem085-mrr128 | `senpai/launchers/C/sglang-mem085-mrr128/` | 0.3396 | 4.01x | `7ccdxfhz` |

- SGLang venv created via `uv venv`; SGLang 0.5.9. First boot hit `libnuma.so.1` import error; resolved by installing `libnuma1 libnuma-dev` via apt.
- Quick = 4 req/profile, dominated by per-request latency rather than concurrent throughput. Full eval expected to lift Scenario C speedup substantially (H100 reference shows SGLang default at 51x on C; on RTX PRO 6000 a 20-50x range is plausible).
- Arm 3 (`max-running-requests 256 --schedule-policy lpm`) skipped per decision tree — Arm 2 did not beat Arm 1 by >10% so direct promotion to full eval.

## 2026-05-27 16:20 — PR #122: vLLM ngram-spec for Scenario B (frieren) — MERGED ✓

- branch: `frieren/vllm-b-decode-tuned` (merged → ib-20260527-latest3-r1)
- hypothesis: vLLM tuned for decode (CUDA graphs + prefix caching + ngram speculative decoding) beats PyTorch baseline on `1/tpot.p50` for Scenario B
- **TERMINAL RESULT — 2.750x speedup over PyTorch baseline, QUALITY PASS**

| Arm | Launcher | 1/tpot.p50 raw | Speedup | W&B |
|---|---|---:|---:|---|
| 1 vllm-cudagraph-prefix | `senpai/launchers/B/vllm-cudagraph-prefix/` | 57.27 | 1.44x | `9w5uw7q3` |
| 2 vllm-cudagraph-block32 | `senpai/launchers/B/vllm-cudagraph-block32/` | 56.55 | 1.42x | `3sl0py0b` |
| 3 vllm-ngram-spec (quick) | `senpai/launchers/B/vllm-ngram-spec/` | 136.90 | 3.44x | `08pnkhgg` |
| 3 vllm-ngram-spec (full, terminal) | `senpai/launchers/B/vllm-ngram-spec/` | **109.346** | **2.750x** | `mqsuiazd` |

- Quick 3.44x → Full 2.750x: speculative decoding acceptance rate under the full 64-request burst at concurrency=1 was somewhat lower than the 4-request probe. Still the clear winner vs Arms 1/2.
- MMLU-Pro: 0.308 vs 0.298 baseline, ratio=1.034, n=500. PASS.
- VRAM: 87,695 MB — exceeds H100 80GB; next launch needs `--gpu-memory-utilization 0.80`.
- validate_result.py: validation_pass=true, terminal_eligible=true.
- Note: frieren completed full B despite advisor steering toward D pivot. The eval completed within the run window (frieren must have queued behind tanjiro, then ran full B after tanjiro's C eval released the slot). Frieren's D pivot was not executed.
- Insight: ngram speculative decoding is the dominant lever on RTX PRO 6000 for output-heavy at c=1. Arms 1/2 (CUDA graphs, prefix caching, block-size tuning) gave only 1.42-1.44x. Speculative decoding multiplied decode throughput ~2.7x by accepting multi-token drafts from n-gram prompt lookups.

## 2026-05-27 16:20 — PR #123: vLLM chunked-prefill for Scenario A (fern) — CLOSED (non-mergeable)

- branch: `fern/vllm-a-chunked-prefill`
- hypothesis: vLLM with chunked prefill and large max-num-batched-tokens beats PyTorch baseline on `1/ttft.p50` for Scenario A
- status: **closed non-mergeable** — quick-only, terminal_eligible=false

| Arm | Launcher | 1/ttft.p50 raw | Speedup | W&B |
|---|---|---:|---:|---|
| 1 vllm-chunked-8k | `senpai/launchers/A/vllm-chunked-8k/` | 2.904 | 1.273x | `evjqnziz` |
| 2 vllm-chunked-16k | committed, not evaluated | — | — | — |
| 3 vllm-chunked-16k-mem092 | committed, not evaluated | — | — | — |

- Arm 1 quick: 1.273x on 1/ttft.p50 (raw 2.904 vs 2.281 baseline). 4/4 requests success.
- Full A (~73 min) did not fit window. GPU slot held by frieren's full B eval.
- Hardware constraint: FP8 KV cache and FlashInfer prefill both blocked on RTX PRO 6000 (`VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_DISABLE_FLASHINFER_PREFILL=1`). Without these, chunked-prefill headroom is minimal and matches H100 vLLM-default (1.25x).
- All 3 launcher files committed to branch at a4c6296. Next launch: run full A with Arm 1 launcher — no implementation work needed.
- Insight: Scenario A's headroom on RTX PRO 6000 without FP8 is ~1.3x. Real gains require H100 with FP8 KV cache. SGLang RadixAttention is worth probing as alternative prefill path.

## 2026-05-27 16:22 — PR #125: vLLM ngram-spec for Scenario D (tanjiro) — CLOSED (no eval)

- branch: `tanjiro/vllm-d-ngram-spec`
- hypothesis: vLLM ngram speculative decoding on Scenario D (4096/2048, c=4) will transfer frieren's B discovery to the general balanced scenario
- status: **closed with no eval** — GPU lease coordination failure; launcher committed but no W&B run

| Arm | Launcher | Speedup | W&B |
|---|---|---:|---|
| 1 vllm-d-ngram5 | `senpai/launchers/D/vllm-d-ngram5/` (committed 32ad1ba) | not evaluated | — |

- GPU slot appeared stale when tanjiro tried to acquire it, but frieren had a live eval running off-lease (or lease had expired but server was still consuming GPU). Coordination failure.
- Launcher committed and preserved for next launch. Full D eval is top priority.
- Insight (theoretical): Scenario D's 2048-token output per request at c=4 should benefit from ngram-spec similarly to B. Quick D is expected to show ≥2x speedup based on B transfer, making it a high-confidence next-launch assignment.
