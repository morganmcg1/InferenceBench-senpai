# SENPAI Research State

- **Time:** 2026-05-25 19:55 UTC
- **Most recent human direction:** none (no GitHub Issues to advisor at boot)
- **Research tag/branch:** ib-20260525-three2-r1
- **Hardware:** 1x RTX PRO 6000 (shakedown, not leaderboard-comparable to H100)
- **Time budget:** 2 hours total, ~1h45m remaining at write time, ~10-15 min reserved for review

## Current research focus
Diversify the first 3 GPU-shared assignments across (a) scenarios with the biggest H100 search/default gap and (b) two engine families. Each student owns a different scenario and engine recipe so their quick probes give the fleet 3 independent signals rather than 3 attempts at the same hill.

- **frieren PR #108 — Scenario A (TTFT prefill), vLLM.** Sweep chunked-prefill ON vs OFF with `max-num-batched-tokens=16384`. H100 ref: SMAC3 4.37x best, vLLM default 1.25x — large headroom.
- **fern PR #109 — Scenario B (TPOT decode), vLLM.** CUDA graphs ON (no `--enforce-eager`), block-size 16 vs 32, prefix caching off. H100 ref: SMAC3 15.23x best, vLLM default 2.25x — biggest absolute headroom.
- **tanjiro PR #110 — Scenario C (high-load throughput), SGLang.** Default vs `--max-running-requests 256 --schedule-policy lpm`. H100 ref: SGLang default 51.12x, SMAC3 best 46.70x (default ≥ search here) — different shape of headroom.

Every PR requires:
- gpu_slot.py for GPU sharing across the 3 students,
- quick probe first → terminal full-eval only on candidates that beat the per-PR default quick result,
- W&B groups `frieren/scA-prefill-sweep`, `fern/scB-decode-sweep`, `tanjiro/scC-throughput-sweep`,
- launchers under `senpai/launchers/<scenario>/<slug>/start_server.sh`,
- runtime_env.sh sourced (Blackwell-safe FlashInfer/FP8-KV defaults).

## Next-up research directions (post-round-1)
1. **Scenario D (general balanced)** — completely unstaffed in round 1. Run a vLLM tuned launcher (chunked-prefill + CUDA graphs + medium concurrency) on Scenario D once any student is idle.
2. **FP8 weight quantization on Mistral-7B-Instruct-v0.3** — separate from FP8 KV cache (which is flagged unstable). Try `--quantization fp8` on a winning launcher; gated by quality_check.pass and ratio ≥ 0.95.
3. **Speculative decoding (n-gram)** — vLLM `--speculative-config` for Scenario A/D where prefill+decode mix benefits from draft-and-verify.
4. **TGI and TensorRT-LLM** — broaden engine portfolio after we have first SGLang and vLLM hardware baselines.
5. **Cross-scenario confirmation of a winning launcher** — once any per-scenario winner is merged, run a single A→D pass under `aggregate/geomean_speedup_over_pytorch` to see if the recipe transfers.

## Open questions for the human research team
None at boot. Will add if a student surfaces an evaluator/infra ambiguity.
