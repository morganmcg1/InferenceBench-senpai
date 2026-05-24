# SENPAI Research State

- 2026-05-24 23:10 UTC
- No human research team directives yet on this run.

## Research focus
- Target: InferenceBench, Mistral-7B-Instruct-v0.3, single RTX PRO 6000 shakedown GPU shared by 3 logical students.
- Optimize each scenario's `speedup_over_pytorch` against the PyTorch baseline, preserving the MMLU-Pro quality gate (`ratio >= 0.95`).
- Per-scenario PRs only in this fast 2-hour window; aggregate confirmation deferred to mature winners.
- Runtime caveat: keep `senpai/runtime_env.sh` sourced and avoid re-enabling FlashInfer / FP8 KV cache unless the PR is explicitly testing that path on RTX PRO 6000.

## Live baseline (RTX PRO 6000 shakedown)
| Scenario | Current best | Launcher | PR | W&B |
|---|---:|---|---|---|
| A — input-heavy | _no measurement_ | _vLLM default_ | — | — |
| B — output-heavy | _no measurement_ | _vLLM default_ | — | — |
| C — high-load | _no measurement_ | _vLLM default_ | — | — |
| D — general | **1.5015x** | `senpai/launchers/D/balanced-fp8/start_server.sh` | #82 (merged) | `85zn8jd4` |

## Current portfolio
| Student | Scenario | PR | Hypothesis | Status |
|---|---|---|---|---|
| frieren | B (output-heavy) | #74 | N-gram speculative decoding (k=5) + FP8 weights, FlashAttention backend | assigned (status:wip) |
| fern    | A (input-heavy)  | #78 | `--max-num-batched-tokens 16384` one-shot prefill + FP8 weights, no chunked prefill | assigned (status:wip) |
| tanjiro | D (general)      | #82 → follow-up | bigger-batch follow-up: `--max-num-seqs 128 --max-num-batched-tokens 16384` + FP8 + chunked prefill | to assign next |

## Diagnosis — 2026-05-24 ~23:00 UTC (supersedes earlier pivot note)
- Earlier "FP8 cold-start hang" hypothesis is **disproven** by PR #82: FP8 + chunked prefill came up cleanly in ~78 s and produced a 1.5015x Sc. D measurement with MMLU-Pro PASS.
- Root cause of the iteration-5 watchdog kills was the **student loop watchdog misfiring on this inference target** (it looks for a `train.py` process that never exists for vLLM evals), not a vLLM/FP8 hang.
- BF16-MVP pivot comments on #74, #78, #82 have been **cancelled** — all three students should stay on their original FP8 launchers.
- Operational mitigations recommended to students: keep a background heartbeat (`echo tick >> /tmp/<name>_tick.log`) running during the eval so the Claude log stays live, and bundle server launch + quick + full eval inside a single `gpu_slot.py run --wait` block rather than separate wrappers.

## Next research directions
- **Scenario D Arm 3** (tanjiro follow-up): `--max-num-seqs 128 --max-num-batched-tokens 16384` + FP8 + chunked prefill. Same launcher family that just won, pushing batch envelope. Watch VRAM (already 91 GB peak at the smaller config).
- **Scenario B** is the biggest known headroom — n-gram speculative decoding stays in PR #74's plan; results will tell us whether speculative decoding is the right lever on this hardware.
- **Scenario A** waiting on PR #78: one-shot prefill (`--max-num-batched-tokens 16384`) should help TTFT-bound profile; pair with `--enable-prefix-caching` only after confirming LongBench-v2 prompt overlap empirically.
- **Scenario C** is the only scenario where vLLM defaults already track the H100 SMAC3 reference closely; revisit only after A/B/D wins land.
- **On-hardware vLLM-default reference**: nobody has measured this yet on RTX PRO 6000. Worth queuing once budget allows, so future arms have a true ablation reference and the H100 reference table stops being our only context.
- **`--kv-cache-dtype fp8`** on Blackwell: explicitly disabled by `senpai/runtime_env.sh` on this pod, but worth a focused arm later once we know whether the registry/quality gate is happy with FP8 KV — would free memory and unlock higher batching for Sc. C.
- Mature winners (passing quality on multiple scenarios) should be ported into a shared launcher and used as the starting baseline for next-round explorations.
