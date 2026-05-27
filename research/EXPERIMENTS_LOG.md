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

## 2026-05-27 15:15 — PR #123: vLLM chunked-prefill for Scenario A (fern) — STALLED

- branch: `fern/vllm-a-chunked-prefill`
- hypothesis: vLLM with chunked prefill and large max-num-batched-tokens beats PyTorch baseline on `1/ttft.p50` for Scenario A
- status: **stalled** — PR has been WIP since 14:44 UTC with zero comments and zero commits beyond the initial assignment. Advisor sent two nudges (15:04 and 15:15). No W&B run logged for this PR group. Cause unknown — could be Claude session hung (started iter 18 at 14:44, SENPAI_CLAUDE_TIMEOUT_SECONDS=3600 means earliest recovery ~15:44 UTC).
- if fern does not reply by 15:35 UTC, PR will be closed to keep the queue clean.
