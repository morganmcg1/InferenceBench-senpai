# SENPAI Research State — `ib-20260525-three1-r1`

- **Date / time:** 2026-05-25 ~17:08 UTC (PR #105 fern MERGED 1.240x Sc. A;
  frieren running full eval but urgently needs to abort+restart with
  --request-limit 24 to fit the 17:34 deadline; tanjiro queued; fern
  assigned PR #107 Sc. C quick)
- **Most recent human directive:** path correction on all 3 round-1 PRs from
  `morganmcg1` (operator). In this packed 3-student pod the student checkouts
  are **per-student**, not shared: students should `cd
  "/workspace/senpai-${STUDENT_NAME}/target"` and call `senpai/gpu_slot.py`
  from there (NOT `/workspace/senpai/target` or `target/senpai/gpu_slot.py`).
  Future PR instructions must reflect this layout.
- **Wall-clock budget (CORRECTED):** 2 hours total, pod created
  **15:34:10 UTC** (not 15:53), so launch ends ~17:34 UTC. About **83 min
  in / ~37 min remaining** as of 16:57 UTC. Frieren caught the original
  miscount in the 16:44 PR comment on #104; advisor re-issued plan at
  16:55-16:57 UTC: fern banks arm 2 only (FP8 arm 3 descoped), frieren
  runs full eval with `--request-limit 24` (~30 min), tanjiro takes a
  brief ~5 min window for tuned-vLLM-D quick. Reserve final 5-10 min for
  advisor merge.
- **Hardware:** 1 RTX PRO 6000 ~96GB GPU shared by 3 students (shakedown,
  not leaderboard-comparable to H100).
- **Pod-watchdog mismatch (NEW):** the group-1 pod entrypoint kills any
  student claude session if "Claude log stale for 601s and no train.py
  process". This already torpedoed frieren's first full-eval attempt at
  16:18:25 UTC. Advisor instructions now ask each student to (a) emit
  periodic stdout inside long `gpu_slot run --wait` bash blocks and (b)
  post a brief PR comment 4-5 min into any long-running eval to reset the
  watchdog timer.

## Current research focus

Establish a tuned, valid launcher per scenario on RTX PRO 6000. The first
round buys information cheaply by running three diverse quick probes across
three different scenarios and two different engine families (vLLM and SGLang)
before committing the scarce full-eval slot.

| Student  | PR  | Scenario | Engine | Status @ 17:08 UTC | Hypothesis |
| -------- | --- | -------- | ------ | ------------------ | ---------- |
| frieren  | 104 | B (output-heavy) | vLLM   | quick 2.89x ngram; running full 64-req eval (started ~16:54); URGENT: advisor posted abort+restart-with-n=24 directive at 17:06 UTC. If frieren aborts ~17:06 and restarts with `--request-limit 24`, ETA terminal ~17:30 UTC | CUDA-graph decode + n-gram speculative (n=3, lookup=5) |
| fern     | 105 | A (input-heavy)  | vLLM | **MERGED** 1.240x (ttft.p50 0.3537s, 128/128, MMLU-Pro PASS). W&B `insngzmf`. FP8 arm 3 staged on branch as round-2 candidate | Tuned vLLM prefill no-prefix. COMPLETED. |
| fern     | 107 | C (high-load)    | vLLM   | NEW PR assigned 17:07 UTC. Quick-only: queue position 3 (after frieren+tanjiro), TTL 600s. If slot frees before 17:30 UTC → quick eval (~2 min at c=32), else post `no_gpu_time` terminal | Default-ish vLLM C launcher: max-num-seqs 64, chunked prefill ON |
| tanjiro  | 106 | D (general)      | vLLM (was SGLang) | Heartbeating blocked, waiter PID 61189 alive; queue position 2. Tuned launcher at `senpai/launchers/D/tanjiro-tuned-vllm-balanced/`. Brief ~5 min window after frieren releases | Tuned vLLM Sc. D: chunked prefill ON, max-num-seqs 32, batched-tokens 8192 |

### Live quick-probe partial results

| PR | Arm | Speedup vs PyTorch | W&B run | Notes |
| -- | --- | -----------------: | ------- | ----- |
| 105 | arm1 (prefix cache ON)  | 1.27x Sc. A | `sm95i0u2` | Close to default-vLLM reference (1.25x). |
| 105 | arm2 (prefix cache OFF) | 1.28x Sc. A | `osss9jxl` | No measurable median improvement vs arm 1. |
| 104 | arm1 (CUDA graphs + no chunked prefill) | 1.43x Sc. B | _none — not logged_ | Real but modest. ITL/TPOT gap suggested non-steady-state overhead. |
| 104 | arm2 (arm1 + n-gram speculative n=3, lookup=5) | **2.89x Sc. B** | _none — not logged_ | **Strongest result so far on this advisor branch.** TPOT 0.0087s vs PyTorch 0.0252s. Quality on quick-mode (16-sample) passes. Needs full eval + W&B log for terminal confirmation. |

### Operational notes

1. **Earlier "orphan GPU" hypothesis was wrong (resolved).** The 93GB VRAM
   at ~15:55 UTC was frieren's arm 2 vLLM server actively running. Slot
   ownership and gpu_slot wrapper were working correctly. The defensive
   orphan-detection comment is still on PR #104 and PR #106 but no orphan
   was present.

2. **Pod-watchdog mismatch (active, NEW).** The group-1 pod entrypoint
   kills student claude sessions if "Claude log stale for 601s and no
   train.py process". This expects training workloads; serving evals look
   idle to it. Frieren's first full-eval was killed at 16:18:25 UTC, the
   GPU was wiped to 0 MiB, and the eval result was lost. Advisor mitigation
   pattern (now in all 3 PRs):

   - Inside long `gpu_slot.py run --wait` bash blocks, emit periodic
     `echo "$(date -u) <progress>"` lines so claude's stdout keeps flowing.
   - Post a brief `STUDENT <name>: full-eval in progress` PR comment 4-5
     min into any long evaluator run — any `gh pr comment` tool call is
     enough to reset the watchdog timer.

   This is a process-level coping mechanism, not a real fix; for future
   pods the entrypoint watchdog would need a serving-aware heuristic.

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

After the current round 1 full-eval confirmations:

1. **Scenario B: bigger speculative tokens or speculative + FP8 weights** —
   if frieren's ngram full-eval confirms ~2.9x, the next bump is
   `num_speculative_tokens=5` or `--quantization fp8` paired with ngram. FP8
   weights have not been tested on this hardware yet; quality gate must hold.
2. **Scenario A with `--quantization fp8` weights** — fern's redirected
   arm 3. Biggest unexplored lever for A on Blackwell tensor cores. Quality
   gate is the risk.
3. **Scenario A: bank default vLLM (~1.28x) as confirmed baseline** if FP8
   arm fails to boot or fails quality. Better to have a measured floor than
   no result.
4. **Scenario D variants** — once tanjiro's tuned-vLLM-D quick lands, the
   next probes are (a) drop chunked prefill to mirror frieren's recipe, and
   (b) add ngram speculative on D (decode budget is half of B but still long).
5. **Scenario C confirmation** with default vLLM — only if there is spare
   GPU time after A/B/D have terminal results. Default vLLM may already be
   near the ceiling on C.
6. **Cross-scenario confirmation** with the round-1 winner (likely frieren's
   ngram launcher applied to A/C/D) — only as a final step inside the
   wall-clock budget.

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
