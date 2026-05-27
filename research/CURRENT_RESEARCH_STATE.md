# SENPAI Research State

- **Timestamp:** 2026-05-27 16:14 UTC (run start)
- **Most recent direction from human researcher team:** none (no open issues).
- **Run setup:**
  - Tag: `ib-20260527-lean1-r1`
  - 3 students (frieren, fern, tanjiro) sharing 1 RTX PRO 6000 (shakedown, not leaderboard-comparable).
  - 2-hour wall-clock budget for the whole research program.
  - Preflight green with PVC assets `rtxpro6000-seed248`.
  - W&B project `wandb-applied-ai-team/inferencebench-senpai`.

## Current research focus

Establish first measured Scenario A / B / C baselines on RTX PRO 6000 while
simultaneously testing one high-headroom serving lever per scenario, so that
the second round can compound on whichever direction shows the biggest gap to
the H100 paper-reference frontier.

Round 1 assignments (all vLLM, diverse knobs, one scenario per student):

| Student | PR | Scenario | Primary lever | Why now |
|---|---|---|---|---|
| frieren | #126 | A (input-heavy / TTFT) | chunked prefill + `max-num-batched-tokens=16384` + FP8 weights | H100 reference shows 1.25x → 4.37x headroom; concurrency=1 means prefill is the only knob. |
| fern | #127 | B (output-heavy / TPOT) | n-gram speculative decoding + FP8 weights | Largest headroom in reference (2.25x → 15.23x); LongBench prompts have structured repetition that n-gram captures. |
| tanjiro | #128 | C (high-load / throughput) | `max-num-seqs=256` + prefix caching + FP8 + 0.92 gpu_mem | H100 default vLLM already 48x; FP8 weights + prefix caching are the realistic remaining throughput levers. |

Scenario D (general) is intentionally **not** in round 1. It is a balanced
geomean of A/B/C levers; once a clear winner emerges in A or B, a follow-up PR
should stack those wins into a D launcher.

## Potential next research directions (queued for round 2+)

Sorted by expected payoff under measured-search discipline. None of these are
assigned yet; pick from this list when reviewing round-1 quick probes.

1. **Scenario B follow-ups if n-gram works:**
   - Add CUDA-graph capture (`--enforce-eager false` is default) + tune `--num-speculative-tokens` over {3, 5, 7}.
   - Try a draft-model speculator (Mistral-7B-Instruct-v0.3 with EAGLE/Medusa head if available pre-quantized).
   - Combine with `--compilation-config` (vLLM's torch.compile pass).
2. **Scenario A follow-ups if chunked-prefill+FP8 works:**
   - Increase `--max-num-batched-tokens` past 16384 (try 32768) to fit the full 8K prompt in one prefill chunk with overhead.
   - Try `VLLM_ATTENTION_BACKEND=TRITON_ATTN` to see if a different kernel handles the long single-prefill better than FlashAttention on RTX PRO 6000.
   - Try AWQ instead of FP8 — quality risk is similar, and on Blackwell AWQ kernels are sometimes faster than vLLM's FP8 W8A8.
3. **Scenario C follow-ups (small-margin tuning):**
   - Compare `--block-size 32` vs default 16 for cache locality at high concurrency.
   - Drop prefix caching (in case LongBench prompts don't actually share prefixes after templating) — measure both with the same quick probe.
   - Lower `--gpu-memory-utilization` to 0.85 if VRAM thrash on shared pod hurts throughput.
4. **Cross-engine probes (only after we have RTX PRO 6000 vLLM measurements):**
   - SGLang for Scenario C — needs per-PR venv via `senpai/create_engine_venv.py`. H100 default SGLang slightly edges vLLM (51.12x vs 48.69x).
   - TensorRT-LLM for Scenario A long prefill — strong reputation for prefill latency on Blackwell, but venv install cost may exceed remaining wall time.
5. **Mature winner confirmation:**
   - Once any scenario has a confirmed RTX PRO 6000 winner, run a cross-scenario `aggregate/geomean_speedup_over_pytorch` confirmation with the best stack-up of A/B/C launchers, to position for a future H100 leaderboard run.

## Operational watchlist

- All 3 students share **one** GPU. Heavy work must serialize via
  `senpai/gpu_slot.py`. Watch for orphaned compute processes after each
  measurement.
- FP8 KV cache is **off-limits** for the FlashAttention backend on this
  hardware (`runtime_env.sh` already disables FlashInfer). Three PRs use FP8
  **weights** only.
- Each launcher must boot cleanly under supervised relaunch — students will be
  asked to demonstrate cold-start after the full eval before merge.
- Reserve the last 10-15 minutes for review/merge/`BASELINE.md` update.
