# SENPAI Research State

- 2026-05-24 (run kickoff)
- No human research team directives yet on this run.

## Research focus
- Target: InferenceBench, Mistral-7B-Instruct-v0.3, single RTX PRO 6000 shakedown GPU shared by 3 logical students.
- Optimize each scenario's `speedup_over_pytorch` against the PyTorch baseline, preserving the MMLU-Pro quality gate (`ratio >= 0.95`).
- Per-scenario PRs only in this fast 2-hour window; aggregate confirmation deferred to mature winners.
- Runtime caveat: keep `senpai/runtime_env.sh` sourced and avoid re-enabling FlashInfer / FP8 KV cache unless the PR is explicitly testing that path on RTX PRO 6000.

## Current portfolio
| Student | Scenario | PR | Hypothesis | Status |
|---|---|---|---|---|
| frieren | B (output-heavy) | #74 | N-gram speculative decoding (k=5) + FP8 weights, FlashAttention backend | assigned (status:wip) |
| fern    | A (input-heavy)  | #78 | `--max-num-batched-tokens 16384` one-shot prefill + FP8 weights, no chunked prefill | assigned (status:wip) |
| tanjiro | D (general)      | #82 | FP8 weights + chunked prefill + `max-num-seqs 64` balanced launcher | assigned (status:wip) |

## Pivot 2026-05-24 ~22:55 UTC — drop FP8 from arm 1
- All three iteration-5 student Claude loops went silent for 42+ min and were watchdog-killed (tanjiro 3030s, frieren 3406s, fern still in flight at this writing).
- Most likely cause: `--quantization fp8` first-load on RTX PRO 6000 vLLM hangs silently inside `gpu_slot.py run --wait` (or HF model download is silently slow).
- All three PRs (#74, #78, #82) have new advisor pivot comments instructing:
  - Arm 1 = BF16 (no quantization) + scenario-relevant levers only.
  - Commit the BF16 launcher file BEFORE acquiring the GPU slot.
  - Run `evaluate.py --quick` first under `gpu_slot.py run --wait --ttl-s 900`.
  - Only layer FP8 in arm 2 after BF16 produces a real measurement.

## Next research directions
- Scenario C is the only scenario where vLLM defaults already match the H100 SMAC3 reference; revisit only after A/B/D wins land.
- If BF16 + scenario-relevant levers passes, then try `--quantization fp8` as a focused arm 2.
- N-gram speculative decoding is the biggest known lever for Scenario B; keep it active in arm 1 even after dropping FP8.
- Investigate `--enable-prefix-caching` value for Scenario A; LongBench-v2 prompts vary so prefix caching may not help — confirm empirically.
- Mature winners (passing quality on multiple scenarios) should be ported into a shared launcher and used as the starting baseline for next-round explorations.
