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

## Round 1 progress (as of 2026-05-24 23:15)

- **PR #79 fern (B):** launcher posted, hit FA3+FP8KV+Blackwell incompatibility, queued for Triton+FP8KV attempt; clear stop-rule and fallback recipe given (drop FP8 KV → specdec+CUDA graphs only). No commits or new comments since 22:44.
- **PR #80 frieren (A):** no commits or comments since 22:44.
- **PR #81 tanjiro (D):** no commits or comments since 22:44.

## Pod deadlock (open, escalated)

- **Symptom:** pod `senpai-ib-20260524-leasefix-r2-group-1-...` stdout last advanced at 22:11:25 UTC (frieren iteration 6 heartbeat). No subsequent heartbeats from any of the 3 students for 60+ min.
- **Diagnostic constraints:** advisor service account lacks `pods/exec` permission; `kubectl top` Metrics API not available. Cannot read `/tmp/inferencebench-gpu-slot.json` from outside the pod.
- **Likely cause:** student Claude session hung while holding (or waiting on) the GPU slot lease. `gpu_slot.py run --wait` default `--wait-timeout-s` is None (unbounded), so a stuck inner command does not auto-release.
- **Escalation:** Issue #90 opened to human team at 22:45 UTC, requesting `kubectl exec` intervention (dump slot file, nvidia-smi, kill stuck vLLM processes). 0 comments after 30 min.
- **Budget at risk:** 2-hour launch window ends 23:53 UTC. Full eval needs ~30 min, so anything beyond ~23:23 UTC start is unlikely to produce a clean result.
- **Advisor next-cycle threshold:** if deadlock persists and remaining budget < 25 min, close PRs #79/#80/#81 as "dead-end on Blackwell pod-deadlock; no W&B run produced for this launch" and end the launch with no merged winners.
