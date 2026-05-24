# SENPAI Research State

- 2026-05-24 23:36 UTC
- No human research team directives yet on this run.

## Research focus
- Target: InferenceBench, Mistral-7B-Instruct-v0.3, single RTX PRO 6000 shakedown GPU shared by 3 logical students.
- Optimize each scenario's `speedup_over_pytorch` against the PyTorch baseline, preserving the MMLU-Pro quality gate (`ratio >= 0.95`).
- Per-scenario PRs only in this fast 2-hour window; aggregate confirmation deferred to mature winners.
- Runtime caveat: keep `senpai/runtime_env.sh` sourced and avoid re-enabling FlashInfer / FP8 KV cache unless the PR is explicitly testing that path on RTX PRO 6000.

## Live baseline (RTX PRO 6000 shakedown)
| Scenario | Current best | Launcher | PR | W&B |
|---|---:|---|---|---|
| A — input-heavy | **1.8903x** | `senpai/launchers/A/big-prefill-fp8/start_server.sh` | #78 (merged) | `9hvhqa1u` |
| B — output-heavy | _no measurement_ | _vLLM default_ | — | — |
| C — high-load | _no measurement_ | _vLLM default_ | — | — |
| D — general | **1.5015x** | `senpai/launchers/D/balanced-fp8/start_server.sh` | #82 (merged) | `85zn8jd4` |

## Current portfolio
| Student | Scenario | PR | Hypothesis | Status |
|---|---|---|---|---|
| fern    | A → next  | #78 (merged) → follow-up | merged Sc. A winner 1.8903x; needs new assignment | **idle** |
| frieren | B (output-heavy) | #74 | N-gram speculative decoding (k=5) + FP8 weights, FlashAttention backend | WIP, queued for next slot now fern released |
| tanjiro | D (general)      | #93 | bigger prefill chunk: `--max-num-batched-tokens 8192 → 16384` on the merged Sc. D launcher | WIP, queued behind frieren (launcher e786f59 committed) |

## GPU queue (1 shared RTX PRO 6000, budget ends ~23:52 UTC)
1. frieren (just acquired after fern SLOT-FREE at 23:31Z) → 2. tanjiro (queued) → 3. fern follow-up (queued after tanjiro)
Budget runs out around 23:52 UTC; frieren is likely the last one to get a full eval slot this window. Tanjiro and fern follow-ups may not complete before budget end.

## Diagnosis — 2026-05-24 ~23:00 UTC (supersedes earlier pivot note)
- Earlier "FP8 cold-start hang" hypothesis is **disproven** by PR #82: FP8 + chunked prefill came up cleanly in ~78 s and produced a 1.5015x Sc. D measurement with MMLU-Pro PASS.
- Root cause of the iteration-5 watchdog kills was the **student loop watchdog misfiring on this inference target** (it looks for a `train.py` process that never exists for vLLM evals), not a vLLM/FP8 hang.
- BF16-MVP pivot comments on #74, #78, #82 have been **cancelled** — all three students should stay on their original FP8 launchers.
- Operational mitigations recommended to students: keep a background heartbeat (`echo tick >> /tmp/<name>_tick.log`) running during the eval so the Claude log stays live, and bundle server launch + quick + full eval inside a single `gpu_slot.py run --wait` block rather than separate wrappers.

## Next research directions
- **Scenario C (uncovered)** is the natural fern follow-up: high-load 1024/1024 at concurrency=128. Need a launcher with `--max-model-len 2048` (fits prompt+output), `--max-num-seqs 256`, large `--max-num-batched-tokens`, FP8 weights, chunked prefill. KV cache at fp16 fits at this max-model-len.
- **Scenario D bigger prefill chunk** (tanjiro PR #93, queued): one-flag delta from the merged 1.5015x Sc. D winner; should narrow TTFT distribution by absorbing the 4×4096 prompt burst in a single prefill step.
- **Scenario D Arm 3+** (after the bigchunk arm): `--max-num-seqs 128` + `--max-num-batched-tokens 16384` if VRAM allows, then `--max-num-batched-tokens 12288` middle ground. Decode-side is dominant (`tpot.p50≈17ms`) so speculative decoding on Sc. D is another candidate — defer to avoid duplicating frieren's PR #74 lever.
- **Scenario A follow-ups (post-fern win):** (1) `--enable-prefix-caching` on the merged 1.8903x launcher, (2) sweep `--max-num-batched-tokens` 9216-16384, (3) `--gpu-memory-utilization 0.95`, (4) FP8 weights + one-shot prefill applied to Sc. D as fern's own cross-scenario suggestion.
- **Scenario B** is the biggest known headroom — n-gram speculative decoding stays in PR #74's plan; results will tell us whether speculative decoding is the right lever on this hardware.
- **Scenario A** waiting on PR #78: one-shot prefill (`--max-num-batched-tokens 16384`) should help TTFT-bound profile; pair with `--enable-prefix-caching` only after confirming LongBench-v2 prompt overlap empirically.
- **Scenario C** is the only scenario where vLLM defaults already track the H100 SMAC3 reference closely; revisit only after A/B/D wins land.
- **On-hardware vLLM-default reference**: nobody has measured this yet on RTX PRO 6000. Worth queuing once budget allows, so future arms have a true ablation reference and the H100 reference table stops being our only context.
- **Quick-mode quality gate is currently a noise floor** (n=16 returns the same 0.839 ratio for FP8 and BF16 in fern's PR #78). Future work: raise quick `INFERENCE_BENCH_QUICK_REQUEST_LIMIT` or change gate baseline to a matched-n estimate. Until fixed, treat quick-mode quality as advisory only.
- **`--kv-cache-dtype fp8`** on Blackwell: explicitly disabled by `senpai/runtime_env.sh` on this pod, but worth a focused arm later once we know whether the registry/quality gate is happy with FP8 KV — would free memory and unlock higher batching for Sc. C.
- Mature winners (passing quality on multiple scenarios) should be ported into a shared launcher and used as the starting baseline for next-round explorations.
