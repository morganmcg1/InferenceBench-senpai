# SENPAI Research State — ib-20260527-guard2-r1

- **Date/time:** 2026-05-27 ~13:43 UTC (round 1 mid-way)
- **Hardware:** RTX PRO 6000 shakedown (1 GPU shared, ~96GB VRAM). Results NOT leaderboard-comparable until repeated on H100.
- **Budget remaining:** ~75 min until 14:59 UTC cutoff
- **Human research team directives:** None

## Round 1 status

| Student | PR | Scenario | Quick result | Full result | Notes |
|---------|-----|----------|--------------|-------------|-------|
| frieren | #119 | A (input-heavy TTFT) | — (blocked on GPU slot 37+ min) | — | Claude iter_13 alive since 13:05:09; no commits/comments. Advisor sent simplification directive: run ONLY A1 once slot frees. |
| fern | #120 | B (output-heavy TPOT) | B1=1.439x speedup (W&B `ru8ph3t8`) | — | Has been holding GPU slot continuously since ~13:12 (30+ min). Posted B1 quick at 13:22. Advisor sent slot-release directive. |
| tanjiro | #121 | D (general balanced) | D1 partial (TTFT 0.193s, TPOT 0.0167s, req/s 0.0376; geomean raw 2.27 reported but conc=1) | — | Claude iter_14 alive since 13:05:51, blocked on `gpu_slot.py run --wait` for D1 full eval. |

## Slot contention (operational issue)

The student pod packs 3 students into one container that serializes Claude sessions. Frieren and tanjiro have been blocked inside `gpu_slot.py run --wait` since 13:05; fern has been holding the slot continuously since ~13:12. Only fern has progressed since iter_15 (13:12).

Mitigation: directed fern to release the slot promptly (#120 comment 4555100125), tanjiro to run D1 full only once acquired (#121 comment 4555100956), and frieren to simplify to A1-only (#119 comment 4555101750). If this doesn't unblock, we will not have terminal Sc.A and Sc.D results this round.

## Decisions and decision tree

- Accept fern B1 (1.439x) as the round-1 baseline candidate for Sc.B if no better arm lands by cutoff.
- For terminal merging, ALL of these must hold per `senpai/validate_result.py`: `validation_pass=true`, `baseline_update_allowed=true`, quality gate ≥ 0.95, real full-eval W&B run ID, exact launcher captured.
- If only quick-eval partials are available at cutoff, no PR can merge as a winner. Document them in `BASELINE.md` as research signals only.
- Round 2 is unlikely given current contention. Focus on getting at least one clean terminal result.

## Round-1 hypotheses (recap)

- **Frieren/Sc.A:** Default vLLM only 1.25x on input-heavy because chunked prefill chunks are too small. A1 = `--max-num-batched-tokens 16384`. Expected 2-4x if it can run.
- **Fern/Sc.B:** Default vLLM 2.25x on output-heavy. CUDA graphs + small max_num_seqs (B1) already at 1.439x quick; n-gram spec decode (B2/B3) on top could compound. Expected 5-10x with spec decode if it runs.
- **Tanjiro/Sc.D:** Mixed profile at conc=4. D1 balanced chunked prefill (8192 batched tokens) — quick at conc=1 not representative.

## Potential next research directions (round 2+ if time)

1. Sc.A with very large chunked prefill (32768 batched tokens) — leveraging 96GB VRAM.
2. SGLang for Sc.A or Sc.C — diversify engine families.
3. Cross-scenario confirmation runs on winning Sc.B launcher.
4. Aggressive `--gpu-memory-utilization 0.94` on shakedown hardware.

## Constraints

- All results shakedown only until H100 repeat.
- FP8 KV + FlashAttention broken on this hardware; do not assign.
- FlashInfer disabled by `runtime_env.sh`.
- 1 GPU shared 3 ways = strict slot serialization is mandatory.
