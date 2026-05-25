# SENPAI Research State — `ib-20260525-three1-r1`

- **Date / time:** 2026-05-25 (launch start)
- **Most recent human directive:** none received yet for this launch (no
  open issues on this advisor branch).
- **Wall-clock budget:** 2 hours, shared across assignment, quick eval,
  advisor review, full eval, and final review window.
- **Hardware:** 1 RTX PRO 6000 ~96GB GPU shared by 3 students (shakedown,
  not leaderboard-comparable to H100).

## Current research focus

Establish a tuned, valid launcher per scenario on RTX PRO 6000. The first
round buys information cheaply by running three diverse quick probes across
three different scenarios and two different engine families (vLLM and SGLang)
before committing the scarce full-eval slot.

| Student  | PR  | Scenario | Engine | Hypothesis |
| -------- | --- | -------- | ------ | ---------- |
| frieren  | 104 | B (output-heavy) | vLLM   | CUDA-graph decode launcher, no chunked prefill, big mem util, prefix caching on, optional n-gram speculative arm |
| fern     | 105 | A (input-heavy)  | vLLM   | Long-prefill launcher, `--max-num-batched-tokens 16384`, no chunked prefill, trimmed `--max-model-len`, optional no-prefix-caching arm |
| tanjiro  | 106 | D (general)      | SGLang | Tuned SGLang `lpm` scheduler + triton attention, with vLLM-default fallback if SGLang fails to import |

Scenario C is intentionally not in the first round — the public reference
shows default vLLM already at 48.69x, close to the tuned ceiling, so it has
low expected multiplier room. Reconsider C only if Scenario A/B/D do not
produce strong quick wins.

## Why these picks

- **Headroom direction.** Public H100 reference shows the largest multiplier
  room between default and tuned launchers on Scenario B (2.25x → 15.23x)
  and Scenario A (1.25x → 4.48x). Even partial transfer to RTX PRO 6000
  should beat the implicit "default vLLM" starting point.
- **Engine diversity.** SGLang on Scenario D is a deliberate hedge against
  collapsing the portfolio to vLLM-only and exposes the `lpm` scheduler as a
  free lever for prompts that share a chat-template prefix.
- **Shared GPU coordination.** All three PRs document the GPU slot protocol
  (`senpai/gpu_slot.py run --wait`) and bound their quick probes so only one
  heavy server/evaluator at a time runs on the shared device.
- **Bounded autonomy.** Each PR has a decision tree: win clearly → ask
  advisor for full eval, neutral → ask for redirect, fail launch → fall back
  to a documented safer recipe. This minimises advisor round trips inside
  the 2-hour clock.

## Potential next research directions (round 2+)

After round-1 quick probes return:

1. **Scenario B with n-gram speculative decoding** — biggest expected TPOT
   win if the prefix-caching/CUDA-graph baseline already beats default.
2. **Scenario A with FlashInfer attention** — only if the default
   `VLLM_ATTENTION_BACKEND=FLASH_ATTN` baseline is the bottleneck on long
   prefill *and* the backend boots cleanly with quality passing.
3. **Scenario D with tuned vLLM** if SGLang underperforms — mirror frieren's
   recipe with c=4-aware `--max-num-seqs`.
4. **Scenario C confirmation** with default vLLM (likely small upside) or
   default SGLang (already 51.12x on H100 reference) — only if there is
   spare GPU time to bank a Scenario C baseline.
5. **FP8 weight quantization** for Scenario B (decode-heavy) if Mistral-7B
   FP8 builds load cleanly on RTX PRO 6000 — guarded by quality gate.
6. **Cross-scenario confirmation** with the round-1 winner, only as a final
   step inside the wall-clock budget if we have a mature single-scenario
   winner.

## Constraints / risks to watch

- **2-hour clock.** Reserve the final 10-15 minutes for terminal reports,
  advisor review, merge, BASELINE.md updates — do not approve a new full
  eval if there is not enough time left.
- **Quality gate.** A speed win that fails MMLU-Pro (>= 0.95 × baseline
  accuracy) is invalid. Frieren's optional speculative arm and any FP8 idea
  must be quality-checked.
- **Hardware-specific kernel paths.** `runtime_env.sh` disables vLLM's
  implicit FlashInfer prefill on RTX PRO 6000 — do not re-enable unless the
  PR is explicitly testing FlashInfer and proves a clean boot. FP8 KV cache
  with `VLLM_ATTENTION_BACKEND=FLASH_ATTN` does not boot on this hardware.
- **GPU slot contention.** Three students, one GPU. Heavy server/evaluator
  commands must go through `senpai/gpu_slot.py run --wait`. Tanjiro is the
  lowest-priority slot owner and will use idle time for setup/inspection.
