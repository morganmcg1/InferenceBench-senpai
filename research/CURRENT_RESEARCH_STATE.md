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

## Next research directions
- Scenario C is the only scenario where vLLM defaults already match the H100 SMAC3 reference; revisit only after A/B/D wins land.
- If FP8 weight quantization passes MMLU-Pro, push to FP8 + KV-FP8 + larger batched-token budget.
- N-gram speculative decoding is the biggest known lever for Scenario B; if accepted-token rate is poor, try short n-gram windows (n=3) before bigger ones.
- Investigate `--enable-prefix-caching` value for Scenario A; LongBench-v2 prompts vary so prefix caching may not help — confirm empirically.
- If quality gate fails, fall back to FP8 weights only or BF16 with all kernel/runtime levers but no quantization.
- Mature winners (passing quality on multiple scenarios) should be ported into a shared launcher and used as the starting baseline for next-round explorations.
