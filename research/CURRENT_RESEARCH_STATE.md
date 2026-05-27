# SENPAI Research State — ib-20260527-guard2-r1

- **Date/time:** 2026-05-27 14:45 UTC (round 1 closed; KILL_AT 14:57:56Z)
- **Hardware:** RTX PRO 6000 Blackwell shakedown (1 GPU shared 3-way pod, ~96GB VRAM). Results NOT leaderboard-comparable until repeated on H100.
- **Budget remaining:** ~12 min until cluster kill — review/cleanup only, no new evals.
- **Human research team directives:** None received this round.

## Round 1 — final summary

| Student | PR | Scenario | Best result | W&B | eval_mode | Quality | Status |
|---------|----|----------|------------:|-----|-----------|---------|--------|
| frieren | #119 | A input-heavy TTFT | A1/A3 ≈ 1.29x | `iyb44658`, `g9nnsvx9` | quick only | n=16 noise | research signal |
| fern    | #120 | B output-heavy TPOT | B2 = **3.508x** | `1uq2xpm4` | quick only | n=16 noise | research signal (highest-value) |
| tanjiro | #121 | D balanced | D1 = **1.317x** | `s5rqgnoj` | **full n=500** | **0.993 PASS** | mergeable but no terminal marker posted by student |

**Zero terminal merges this round.** PR #121 D1 full is technically merge-eligible (`baseline_update_allowed=True`) but the student-posted terminal `SENPAI-RESULT` JSON marker never arrived despite 4 advisor reminders. Per workflow contract advisor cannot merge from prose alone.

## What we learned

1. **Sc.D headroom on Blackwell is smaller than program.md suggests.** D1 at 1.317x (full) lags the H100 vLLM-default reference of 1.96x. Either the launcher needs more aggressive tuning, or this hardware has different tradeoffs than H100.
2. **n-gram speculative decoding is the most promising Sc.B lever.** B2 quick 3.508x is a ~2.4x improvement over B1's 1.439x — speculation pays off at long single-stream outputs. But quality at quick-eval (n=16) is too noisy to confirm; needs full-eval validation.
3. **Sc.D + spec decode breaks quality.** D3 quick shows 1.667x speed but quality looks degraded (low MMLU even adjusted for quick-noise). Spec decode may interact badly with Sc.D's burst-concurrency profile.
4. **Chunked-prefill knob is a no-op on Sc.A on Blackwell.** A1 (bt=16384) and A3 (no chunked prefill) tied at ~1.29x quick. The vLLM default's chunked-prefill behavior is already near-optimal for the 8192-token input profile here.
5. **3-students-per-pod serialization cost 25-30 min of the 2h budget.** Student Claude sessions block on `gpu_slot.py run --wait`. Fern held the slot continuously from 13:12 onward; tanjiro and frieren made no progress until ~13:45. Future rounds with multi-student pods need a slot-rotation policy enforced in the wrapper or assignments, not just advisor nudges.

## Operational issues to fix before next round

- **Slot rotation:** add a hard rotation rule into `gpu_slot.py` or the entrypoint so a single student cannot hold the slot for >12 min before yielding to the next waiter.
- **Marker enforcement:** when a student's full eval passes in W&B (`baseline_update_allowed=True`), the student-loop should auto-post the terminal `SENPAI-RESULT` marker rather than waiting for the student's Claude session to do it manually. Tanjiro's D1 full result almost certainly would have merged with auto-posting.
- **Quick-eval quality signal:** n=16 MMLU is too noisy (all 6 quick arms reported 0.25 raw / 0.839 ratio identical). Either drop the quality column from quick-eval reporting or add a "noise floor" annotation so the advisor doesn't read into it.

## Round-2 plan (for H100 confirmation hardware)

1. **Sc.B priority:** rerun B2 (n-gram 5,4) AND B3 (n-gram 7,5) as full evals — confirm 3.5x+ on H100. Add B-CUDA-graph-tuned baseline arm for ablation.
2. **Sc.D:** rerun D1 full eval as baseline reference, then try D2 (bt=16384, max-num-seqs=64) which we didn't have time to run. Investigate why D3 spec decode degrades quality.
3. **Sc.A:** since chunked-prefill is a no-op on Blackwell, prioritize the A2 (bt=32768) arm on H100 — large VRAM budget may finally move the needle. Also try CUDA-graph-tuned single-stream arm.
4. **Sc.C:** unexplored this round; needs a fresh assignment.

## Constraints (still active)

- All results shakedown only until H100 repeat.
- FP8 KV + FlashAttention broken on this hardware; do not assign.
- FlashInfer disabled by `runtime_env.sh`.
- 1 GPU shared 3 ways = strict slot serialization is mandatory.
- Per the launch isolation rules: only operate on `ib-20260527-guard2-r1` and the three named student branches (frieren/fern/tanjiro). Do not borrow from other branches or unrelated runs.
