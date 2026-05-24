# SENPAI Research State — `ib-20260524-leasefix-r2`

- **Date/time:** 2026-05-24
- **Most recent human research direction:** None — fresh launch, no GitHub Issues from the team.
- **Hardware regime:** RTX PRO 6000 Blackwell shakedown, NOT H100 leaderboard. Results are shakedown evidence only until repeated on H100.
- **Budget:** 2-hour wall clock for the entire program (assignment → evaluation → review → merge → next round). 3 students sharing 1 GPU through `senpai/gpu_slot.py`.

## Current research focus

Round 1 of the InferenceBench launch. Goal: get one solid valid launcher per scenario (B, A, D) above the PyTorch baseline before round 2, then compound improvements.

Scenario C is intentionally deprioritized this round: vLLM defaults already beat the SMAC3 search reference there on H100 (48.69x vs 46.70x), so the marginal value of tuning C is much lower than tuning B (2.25x → 15.23x gap on H100 reference).

## Round 1 assignments (in flight)

| PR | Student | Scenario | Mechanism |
|---|---|---|---|
| [#79](https://github.com/morganmcg1/InferenceBench-senpai/pull/79) | fern | B (output-heavy) | n-gram spec decoding 5 tok + FP8 KV cache + CUDA graphs + FLASH_ATTN |
| [#80](https://github.com/morganmcg1/InferenceBench-senpai/pull/80) | frieren | A (input-heavy) | long-context prefill: block_size=32 + CUDA graphs + prefix caching + FLASH_ATTN |
| [#81](https://github.com/morganmcg1/InferenceBench-senpai/pull/81) | tanjiro | D (balanced) | chunked prefill 8192 + spec decoding 3 tok + CUDA graphs + max_num_seqs 64 |

All three start from `src/starting_points/vllm_running/start_server.sh`, source `senpai/runtime_env.sh`, and coordinate the heavy workload through `gpu_slot.py`.

## Potential next research directions (round 2+)

1. **Compound the best round-1 lever across scenarios.** If FP8 KV cache passes quality on B, port it to D. If chunked-prefill at 16384 tokens beats default on A, port it to D.
2. **SGLang or TGI on the worst-performing scenario.** vLLM is one engine — for scenarios where vLLM round-1 underperforms, a one-PR SGLang or TGI launcher may close the gap quickly.
3. **Bigger speculative-decode window** when the n-gram acceptance rate is high (round-1 evidence will show this in TPOT vs raw decode rate).
4. **TensorRT-LLM** compile-once launcher for the winning scenario (high upfront cost, big payoff).
5. **Attention-backend ablation on Blackwell:** FLASH_ATTN vs FLASHINFER vs TRITON_ATTN when the runtime_env default is overridden, only for scenarios where the round-1 server boots cleanly.
6. **Scenario C touch-up:** small-budget run to confirm vLLM defaults still win and protect against regression once a winner is being chosen for cross-scenario confirmation.
7. **GPU-memory-utilization sweep** (0.92 → 0.95) if FP8 KV cache opens KV-block headroom.

## Open risks

- **Blackwell + FP8 KV cache is path-restricted.** Confirmed via fern PR #79: FA3 refuses (SM≠9), FlashInfer JIT errors on CUDART/nvcc mismatch. Only TRITON_ATTN supports FP8 KV here. See BASELINE.md "Hardware constraints" section.
- **Blackwell FlashInfer/FP8 stability.** Runtime env disables FlashInfer prefill by default. Students must NOT re-enable it without measured stability.
- **Quality gate from speculative decoding or FP8 KV.** Spec decoding should be safe (verification); FP8 KV is the more likely culprit on Triton path. Fallbacks are written into each assignment.
- **GPU slot contention.** Three students, one GPU. Observed working — tanjiro holds lease for D, fern queued via `--wait` (correct behavior). Watch for stale leases.
- **GitHub rate-limit risk** — keep `gh` calls deliberate; lean on the W&B project and the GPU slot for high-frequency state.

## Round 1 progress (as of 2026-05-24 23:27)

- **PR #81 tanjiro (D):** **MERGED 23:20:53 UTC.** 1.708x speedup, MMLU-Pro 0.980 PASS. W&B `42ajz9lf`. First round-1 winner. Recipe: chunked-prefill 8192 + specdec ngram 3 tok + max_num_seqs 64 + block_size 32 + FLASH_ATTN.
- **PR #79 fern (B):** still draft/wip, no commits since 22:00:22, no comments since 22:44. fern's iteration 4 Claude session started 22:00:37 — possibly still running like tanjiro's was (his ran 4940s = 82 min before producing the winner).
- **PR #80 frieren (A):** still draft/wip, no commits since 22:00:34, no comments since 22:44. frieren's iteration 6 Claude session started 22:11:25 — same uncertainty as fern.
- **PR #96 tanjiro (D compound):** new assignment 23:26 UTC. Bump `num_speculative_tokens=5, prompt_lookup_max=6` on top of #81's winning recipe. Explicit instruction to not preempt #79/#80 leases and to skip full eval if start > 23:30 UTC.

## Lesson learned: pod stdout buffering vs deadlock

Earlier (22:11 → 23:23 UTC) the advisor mis-read pod stdout silence as a deadlock and opened Issue #90 to humans. It was wrong — Claude student sessions can run far longer than the typical 350s, and the entrypoint script's iteration heartbeats only print between iterations. tanjiro's iteration ran 4940s (82.3 min) doing real work the entire time. **Rule going forward:** treat the PR's commits/comments and the W&B project as authoritative for student activity. Do not assume pod-stdout silence means stuck — Claude sessions can legitimately run an hour or longer. Pod restart / Claude crash are visible as `Claude exited code=N` lines, so absence of those is evidence the session is still running.

## Time remaining

- Launch ends 23:53 UTC. As of 23:27 ≈ 26 min remaining.
- Full eval requires ~30 min minimum (3-5 min cold start + ~25 min eval). Quick eval ~2-3 min.
- If fern/frieren produce terminal results before budget end, merge winners best-first.
- PR #96 tanjiro is unlikely to produce a full eval before budget end; quick eval may be possible if GPU lease frees up immediately.
