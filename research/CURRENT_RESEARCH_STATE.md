# SENPAI Research State — ib-20260524-ready-r2

- **As of:** 2026-05-24 (advisor boot, T+0 of the 2-hour program)
- **Most recent direction from the human research team:** Operator note — RTX PRO 6000 scoring assets prepared at `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`. This is a shakedown run, NOT a leaderboard claim. Hydrate/check those assets before reporting `speedup_over_pytorch`. No GitHub issues from the human team at boot.
- **Mode:** RTX PRO 6000 shakedown — Blackwell ~96GB, 1 shared GPU across 3 students, 2-hour total program budget.
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`.

## Current research focus and themes

Optimize one InferenceBench scenario per PR, paper-facing metric is `scenario/<X>/speedup_over_pytorch`. The first round attacks the three scenarios with the largest headroom over the unoptimized vLLM default (per the 2026-05-21 H100 reference snapshot):

| Slot | Student | Scenario | Hypothesis core | Reference ceiling | PR |
|---|---|---|---|---:|---|
| 1 | r2-frieren | B (output-heavy, TPOT) | n-gram speculative decoding + FP8 KV + CUDA graphs | 15.23x ceiling / 2.25x default | #43 |
| 2 | r2-fern | A (input-heavy, TTFT) | FlashInfer attention + FP8 KV + tuned prefill batch | 4.37x / 1.25x | #44 |
| 3 | r2-tanjiro | D (general, balanced) | chunked prefill + n-gram speculative + FP8 KV | 5.69x / 1.96x | #45 |

Scenario C (high-load throughput) is intentionally deferred this round — public reference shows vLLM default already at 48.69x vs SMAC3 46.70x, i.e. essentially saturated, so it offers the least headroom for a first-round measured experiment.

**GPU coordination:** 1 GPU, 3 students. PRs explicitly assign slots in time order (B → A → D) and each student must wait for the previous `SLOT-FREE` signal before launching their server. Heavy GPU work never runs concurrent. Students do prep work (launcher scripts, task workspace, smoke checks of CPU-only paths) while another student holds the GPU.

**Cross-cutting bets in this round:**
1. **FP8 KV cache** is set in all three launchers — cheap memory win on Blackwell with negligible quality impact in past evidence.
2. **CUDA graphs (no `--enforce-eager`)** in all three — every per-step kernel-launch overhead saving compounds across thousands of decode steps in B and D.
3. **n-gram speculative decoding** in B and D — no draft model required, only relevant where output length is large enough to amortize verify cost (B 8192 tok, D 2048 tok). Skipped for A (1024 tok output, prefill-dominated).
4. **Attention backend variation:** Fern probes FLASHINFER for A (long prefill); Frieren and Tanjiro stay on FLASH_ATTN for stability. Fern has an explicit 5-minute FLASH_ATTN fallback rule so we don't lose her slot to backend debugging.

## Potential next research directions and themes (after first round results land)

1. **Compound the wins.** Whichever knobs land cleanly in round 1 (FP8 KV, CUDA graphs, attention backend, speculative tokens) merge into BASELINE.md and become the starting point for round 2 on other scenarios.
2. **Scenario C dedicated arm.** The reference shows SMAC3 actually loses to vLLM default on C (46.70x vs 48.69x). A simple "default + FP8 KV + CUDA graphs" launcher may be the right C entry — narrow, low-risk arm that documents whether C is genuinely saturated on this Blackwell hardware or whether the public number is misleading.
3. **Aggressive speculative tuning.** If Frieren's n-gram speculator lands a clear win on B, sweep `num_speculative_tokens` and `prompt_lookup_max` for the bigger lever (5/4 vs 7/4 vs 3/3) — accept-rate is workload-dependent.
4. **FP8 weight quantization.** All three round-1 launchers keep weights in their default dtype. If quality holds at FP8 KV, an FP8 weight launcher (`--quantization fp8`) is the next obvious memory-pressure / compute-saturation lever, especially for Scenario C throughput.
5. **SGLang or TGI second engine probe.** Reference snapshot only shows vLLM after HPO. SGLang's `lpm` schedule and torch.compile path are an unexplored arm in our SENPAI history on this hardware. Hold this until vLLM levers are mostly exhausted.
6. **Cross-scenario confirmation runs.** Any single-scenario winner that doesn't visibly regress quality is a candidate for an A-D confirmation pass to compute `aggregate/geomean_speedup_over_pytorch` — required before any leaderboard claim, even on shakedown.
7. **Tail-latency hardening.** All round-1 launchers chase p50. If the chosen knobs blow up p99 or `failure_rate`, the next round should add tail-aware levers (lower `max_num_seqs`, prefix-cache, scheduler policy) and re-measure.

## Constraints to keep in mind

- 2-hour total budget covers assignment + quick eval + advisor review + final validation + cleanup. Treat every minute as scarce.
- All work goes under `senpai/launchers/<scenario>/<slug>/` and `senpai/` helpers. `src/eval/`, `src/baselines/`, `src/run_task.sh`, `src/starting_points/`, scenario files, and `containers/` are protected.
- Quality gate is hard: MMLU-Pro observed >= 0.95 * baseline (baseline = 0.298, so observed must be >= 0.283).
- This is RTX PRO 6000 shakedown — winners must NOT be reported as leaderboard-comparable until repeated on H100.
