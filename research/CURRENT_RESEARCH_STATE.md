# SENPAI Research State — `ib-20260528-scen-d-r1`

- **Updated:** 2026-05-28 17:47 UTC (start gate 16:36:57 UTC, budget end ~18:36 UTC, ~49 min left; review at 18:20 UTC, latest full-eval start ~17:55 UTC)
- **MAJOR FINDING:** Scenario D speed eval runs at concurrency=1 because `src/eval/inference/runner.py:1159` defaults profile-level concurrency to 1 when scenario.json has only a `profile` dict (no `profiles` list). Scenario D's profile is `{"name":"burst","pattern":"burst"}` with no concurrency. The PyTorch baseline_metrics.json also only has the c=1 burst profile, so speedups vs baseline are apples-to-apples. Wider-batch / chunked-prefill / prefix-cache levers are dormant; per-token decode levers (speculative decoding, quantization, decoder kernels) dominate.
- **Launch budget:** ~2 hours, single Scenario D.
- **Most recent human directive:** none in this launch (no open team issues).
- **GPU topology:** 1 RTX PRO 6000 Blackwell, shared by 2 logical students (scen-d-frieren, scen-d-fern). Coordinate via `senpai/gpu_slot.py`.

## Current focus

Round 1 complete for frieren (merged, 1.2919x). Round 2 (speculative decoding) now active via fern.

**Current best:** PR #155 merged — vLLM tuned, **1.2919x speedup_over_pytorch**, quality 1.02 (PASS, n=500), 96/96 success, W&B `lyzskqbu`.

Slot is FREE as of 17:45 UTC. Fern has been directed to take it immediately for n-gram speculative decoding pivot.

## Active portfolio

| PR | Student | Engine | Latest signal | Next step |
|---|---|---|---|---|
| #155 | scen-d-frieren | vLLM tuned | **MERGED** at 17:46 UTC — full eval **1.2919x**, quality 1.02, 96/96. Launcher: `senpai/launchers/D/frieren-vllm-balanced/start_server.sh`. | Stood down. |
| #157 | scen-d-fern | vLLM-spec NGRAM (pivot) | Arm 1 LPM quick **1.168x** at c=1. Pivoted to speculative decoding. Go directive posted at 17:46 UTC. Slot free since 17:45. | Take slot immediately. Quick probe: vLLM + `--speculative-config '{"method":"ngram","num_speculative_tokens":5,...}'`. If quick ≥ 1.3x → full. Terminal SENPAI-RESULT by 18:15 UTC. |

### Concurrency-1 caveat (now load-bearing for the launch)

The scenario.json `concurrency: 4` field is NOT read by the speed evaluator. `runner.py:1159` falls back to `config.get('profile', ...)` and Scenario D's profile dict has no `concurrency`, so default=1. This means BOTH quick AND full eval run at c=1 for Scenario D. Wider-batch / chunked-prefill / prefix-cache levers are dormant at c=1 by construction. The c=1 PyTorch baseline (38.21 tok/s gen, 0.0382 req/s) is the real comparison.

This is not a bug we should fix — `scenario.json` is a protected benchmark file. We optimize for the actual c=1 scoring function.

### What actually moves the metric at c=1

1. **n-gram (lookup) speculative decoding** — biggest single lever at c=1, no draft model needed. vLLM: `--speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":3,"prompt_lookup_min":2}'`. SGLang: `--speculative-algorithm NGRAM ...`.
2. **Decoder kernel choice** — SGLang's triton kernel gave +59% gen throughput vs PyTorch; vLLM's FLASH_ATTN gave +46%. Both already on the board.
3. **CUDA graphs ON** — already enabled in merged recipe.
4. **FP8 quantization** — deferred; quality risk.

## What we learned from round 1

- **vLLM tuned (no spec):** 1.2919x at c=1. TPOT p50 1.48x faster, p90 2.06x. TTFT p99 regresses (chunked-prefill interleaving). Quality marginally above baseline.
- **SGLang LPM triton:** 1.168x at c=1 (quick only). +59% gen throughput — the triton decode kernel is fast, but TTFT is slower and there's no speculative acceleration.
- **c=1 is the real workload.** All measurements are apples-to-apples vs the c=1 PyTorch baseline.

## Remaining budget / risks

- Hard wall: 18:36 UTC. Review window: 18:20 UTC. Fern's full eval must complete by 18:15 to leave 5 min for validate+finalize+review.
- If fern's speculative quick < 1.3x, the spec config may be wrong (temperature mismatch, acceptance rate near zero). Should try `num_speculative_tokens 3` once.
- If spec quick ≥ 1.3x but fern can't start full by 17:55, we still have a merged baseline at 1.2919x — the launch is not empty-handed.
- FlashInfer is still disabled — keep it that way.
