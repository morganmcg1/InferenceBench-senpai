# SENPAI Research State — `ib-20260525-three1-r1`

- **Date / time:** 2026-05-25 ~17:27 UTC (PR #105 fern MERGED 1.240x Sc. A;
  PR #106 tanjiro CLOSED informational 1.246x Sc. D quick — launcher banked
  for round-2 full-eval; PR #107 fern Sc. C CLOSED descoped/no-slot;
  PR #104 frieren n=12 authorized at 17:21:32, no ack yet — heartbeat
  prompted at 17:27; ETA terminal ~17:36-17:38 UTC ~2-4 min past deadline)
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

| Student  | PR  | Scenario | Engine | Status @ 17:27 UTC | Hypothesis |
| -------- | --- | -------- | ------ | ------------------ | ---------- |
| frieren  | 104 | B (output-heavy) | vLLM   | n=12 authorized 17:21:32 after queue contention from stale fern wrapper. No ack yet at 17:27 (5.5 min). Heartbeat prompt posted. ETA terminal ~17:36-17:38 if running. Speed eval at `--request-limit 12`, quality at n=500. | CUDA-graph decode + n-gram speculative (n=3, lookup=5) |
| fern     | 105 | A (input-heavy)  | vLLM | **MERGED 17:03** 1.240x (ttft.p50 0.3537s, 128/128, MMLU-Pro PASS). W&B `insngzmf`. FP8 arm 3 staged on branch as round-2 candidate. | Tuned vLLM prefill no-prefix. COMPLETED. |
| fern     | 107 | C (high-load)    | vLLM   | **CLOSED 17:21:53** — slot contention (stale wrapper) + wall clock + low expected room (H100 default 48.69x already near tuned ceiling). Launcher at `senpai/launchers/C/fern-vllm-baseline-c/` staged for next launch. | Default-ish vLLM C launcher (deferred). |
| tanjiro  | 106 | D (general)      | vLLM | **CLOSED 17:25 informational** 1.246x quick (vs 1.241x default-vLLM-D quick floor). Not BASELINE-eligible (quick eval, not full). Two launchers banked: `tanjiro-tuned-vllm-balanced/` (commit `efdc787`) and `tanjiro-fallback-vllm-default-d/` (commit `6dca712`). Round-2 priority: pair tuned-D with frieren's ngram speculative config. | Tuned vLLM Sc. D: chunked prefill ON, max-num-seqs 32, batched-tokens 8192. COMPLETED (informational). |

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

Round 1 outcome summary:
- **Sc. A floor banked: 1.240x** (fern PR #105 full-eval, MMLU-Pro PASS).
- **Sc. B in flight: PR #104** frieren n=12 ngram speculative — quick was
  2.89x; if n=12 full-mode lands cleanly the lift is real and merge-worthy.
- **Sc. C: no result this round** — fern PR #107 closed (slot contention +
  H100 reference shows default already 48.69x, low room).
- **Sc. D: 1.246x quick informational** (tanjiro PR #106 closed), tuned
  launcher banked for round-2 full eval.

Round-2 priorities (ranked):

1. **Scenario B confirmation at full mode** — if frieren's PR #104 lands a
   clean n=12 terminal, the next launch should immediately full-eval the
   same launcher (`senpai/launchers/B/frieren-tuned-vllm-decode-ngram/`) at
   the full 64-request budget to bank the production floor. If n=12 quality
   fails, drop to `num_speculative_tokens=2` and re-run.
2. **Scenario D tuned full eval + ngram pairing** — pick up
   `senpai/launchers/D/tanjiro-tuned-vllm-balanced/` (commit `efdc787`) and
   run a full 96-req c=4 burst eval. Then pair with frieren's ngram
   speculative config (`{"method":"ngram","num_speculative_tokens":3,
   "prompt_lookup_max":5}`) — the highest-EV cross-scenario follow-up.
3. **Scenario A with `--quantization fp8` weights** — fern's deferred
   arm 3 launcher at
   `senpai/launchers/A/fern-tuned-vllm-prefill-noprefix-fp8/` is staged on
   the merged branch. Biggest unexplored lever for A on Blackwell tensor
   cores. Quality gate is the risk.
4. **Scenario B: bigger speculative tokens** — if frieren confirms ~2.9x,
   try `num_speculative_tokens=5` or `num_speculative_tokens=7` to test
   whether the lift saturates or keeps climbing on Mistral-7B.
5. **Scenario C floor measurement** — pick up fern's staged
   `senpai/launchers/C/fern-vllm-baseline-c/` launcher. One quick eval is
   enough to establish the RTX PRO 6000 floor on Sc. C (H100 default is
   already at 48.69x — Sc. C is the lowest-headroom scenario per the
   reference table).
6. **SGLang LPM hypothesis** — only after a fresh container with
   `sgl-kernel >= 0.3.20` matched to `sglang 0.5.12.post1` is available.
   Blocked operationally, not by research direction.
7. **Cross-scenario confirmation with round-1 winners** — apply the
   ngram speculative recipe (if confirmed on Sc. B) to A/C/D as a single
   unified launcher.

## Operational lessons for next launch

1. **Watchdog mismatch** must be fixed at the pod-entrypoint level. The
   "Claude log stale for 601s and no train.py process" heuristic is the
   wrong test for serving workloads. Recommend the entrypoint check for
   active `vllm` / `sglang` / `evaluate.py` processes OR a heartbeat file
   before killing claude.
2. **gpu_slot.py queue contention** is a real issue when student wrappers
   stack up. PR-close does NOT kill local waiter PIDs. Future advisor
   instructions should explicitly tell students to `kill` their waiter pid
   when a PR is closed, or `gpu_slot.py` should respect a PR-closed signal.
3. **Wall-clock math** needs to start from pod creation time
   (`kubectl get pod -o jsonpath='{.status.startTime}'`), not from the
   first advisor commit. Frieren correctly caught a 19-minute miscount
   that nearly cost the launch.
4. **Quick vs full eval contract** worked — tanjiro's 1.246x quick was
   correctly held back from BASELINE.md. The contract is doing its job.

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
