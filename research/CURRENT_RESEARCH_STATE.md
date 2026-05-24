# SENPAI Research State — ib-20260524-ready-r3

- **Date:** 2026-05-24
- **Hardware mode:** RTX PRO 6000 shakedown (not leaderboard-comparable)
- **Time budget:** 2-hour research clock total
- **Advisor branch:** `ib-20260524-ready-r3-advisor`
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`

## Most recent direction from human researcher team

No GitHub Issues from the human researcher team at boot. The operator did
confirm the prepared RTX PRO 6000 scoring assets at
`/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248` pass
hard A–D preflight; this is shakedown evidence, not an H100 leaderboard claim.

## Round 1 status (as of 2026-05-24 07:50 UTC)

- **Slot 1 / PR #37 (r3-frieren, scenario B):** ACTIVE. iter 17 timed out at
  07:22 (Claude session, no progress lost — launcher local). Hard deadline
  07:42 averted by iter 19 progress comment at 07:40:21 (commit `9268663`
  pushed; live server reused from iter 17, PID 19492). Currently running
  reduced-N strategy: `--request-limit 32` (half the 64-req burst profile)
  + `INFERENCE_BENCH_QUALITY_MMLUPRO_N=64` (vs 500), started 07:44 to fit
  ~30–40 min slot. Advisor accepted this for the RTX PRO 6000 shakedown but
  required clear caveat labeling in the SENPAI-RESULT JSON
  (`caveats:["request_limit_32_of_64","mmlupro_n_64_of_500"]`).
- **Slot 2 / PR #38 (r3-fern, scenario A):** WAITING. Launcher pushed
  (`b837ca1`). Told to stand down on 07:42 swap (averted) and continue
  queue-wait for `SLOT-FREE` on PR #37. Estimated wait ~25–35 min.
- **Slot 3 / PR #40 (r3-tanjiro, scenario D):** WAITING. Launcher pushed
  (`6762651`). Queue position unchanged.

## Current research focus

Round 1 opens a three-way orthogonal probe of the highest-headroom scenarios
on the public reference snapshot (H100, public InferenceBench numbers). Each
student owns one scenario and one fresh vLLM launcher recipe stored under
`senpai/launchers/<scenario>/<slug>/start_server.sh`. The PRs only modify
`senpai/` paths; all benchmark code (`src/eval/`, `src/run_task.sh`,
`agents/`, `containers/`) is untouched. The 3 students share one RTX PRO 6000
GPU, so the round is serialized via a GPU queue with a `SLOT-FREE` signal.

| Slot | PR | Student | Scenario | Hypothesis | Why |
|---:|---:|---|---|---|---|
| 1 | #37 | r3-frieren | B (output-heavy / TPOT) | vLLM + FP8 KV + `--max-num-seqs 1` + chunked-prefill OFF + prefix-caching OFF + block-size 16, CUDA graphs ON | Largest reference headroom (~2.25x → ~15x); decode-bound; KV bandwidth + scheduler overhead are the bottlenecks |
| 2 | #38 | r3-fern | A (input-heavy / TTFT) | vLLM + `--max-num-batched-tokens 16384` + chunked-prefill OFF + prefix-caching OFF + FP8 KV, CUDA graphs ON | Big reference headroom (~1.25x → ~4.5x); single 8k prefill in one forward pass |
| 3 | #40 | r3-tanjiro | D (general balanced / geomean) | vLLM + `--max-num-seqs 8` + chunked-prefill ON + prefix-caching OFF + FP8 KV, CUDA graphs ON | Medium reference headroom (~1.96x → ~5.69x); concurrency 4 benefits from overlap |

Scenario C (high-load throughput) is intentionally deprioritized for round 1 —
vLLM default already sits near the SMAC3 reference on H100 (~48.7x vs ~46.7x);
small headroom is not worth the only-one-GPU serialization cost in round 1.

## Decision rules already encoded in this round

- All launchers use `--kv-cache-dtype fp8` only (not full-weight FP8) so a
  failed MMLU-Pro gate clearly points to the KV-cache lever rather than
  weight quantization, and so the first round stays low-risk to quality.
- All launchers use `--block-size 16`, `--gpu-memory-utilization 0.92`,
  `--max-model-len 16384` to bound memory and isolate scheduler effects.
- All launchers run as a foreground `exec` (no `nohup`/`setsid`/`&`) per
  `target/program.md` server contract.
- Cross-PR coordination is explicit: each PR body contains the GPU queue
  order, and an advisor comment on each PR pins the slot mapping
  (#37 → #38 → #40) and the `SLOT-FREE` signal.

## Potential next research directions (round 2+ candidates)

Conditional on round 1 results:

1. **If KV-FP8 holds quality on B / A / D**: try `--quantization fp8`
   (full-weight FP8) on the winning launcher to layer another speedup.
2. **n-gram speculative decoding** on B / D: `--speculative-config` with 3–7
   draft tokens to cut TPOT another factor on long-decode workloads.
3. **Attention backend swap**: probe `VLLM_ATTENTION_BACKEND=FLASHINFER` and
   `TRITON_ATTN` on Blackwell — backend choice matters more on RTX PRO 6000
   than the public H100 references suggest.
4. **Block size 32** as a follow-up arm on a scenario-winning launcher to
   trade KV fragmentation for fewer block-table ops.
5. **Scenario C dedicated probe** if a student frees up: high `--max-num-seqs`
   (256, 512), `--enable-prefix-caching` on poisson/constant profiles where
   tail-of-request reuse exists, and aggressive `--max-num-batched-tokens`.
6. **SGLang or TGI**: only after vLLM headroom is exhausted on a given
   scenario, since engine swap is a bigger move and adds container/runtime
   debugging cost.
7. **Cross-scenario confirmation run**: only when a single launcher beats
   the baseline on every scenario, run an A–D sweep and report
   `aggregate/geomean_speedup_over_pytorch`.

## Open risks and watch-items

- **Blackwell/FlashInfer/vLLM startup** quirks — students must
  `source "$PROBLEM_DIR/senpai/runtime_env.sh"` before each launch. If a
  smoke test fails before any eval, advisor should expect a `pending_arms`
  partial-result comment rather than radio silence.
- **Quality gate sensitivity to FP8 KV** — Mistral-7B-Instruct usually
  retains MMLU-Pro accuracy with FP8 KV, but RTX PRO 6000 FP8 paths
  haven't been measured by SENPAI yet on this advisor branch.
- **Shared-pod measurement contamination** — if any student uses broad
  `pkill` they can corrupt another student's measurement. Each assignment
  body explicitly forbids `pkill -f vllm` / `pkill python`.
- **2-hour clock** — slot 3 is the most at-risk; if slot 1 or 2 overruns,
  r3-tanjiro may not get a full-eval window. Acceptable: tanjiro returns a
  quick-eval probe with `pending_arms:true` and waits for round-2 advisor
  steering.
- **Reduced-N comparability** — slot 1's `--request-limit 32` measurement
  cannot be directly compared to the full 64-req burst profile baseline in
  `inference_scenario_b_output_heavy/baseline_metrics.json`. The
  `speedup_over_pytorch` value frieren reports will be `1/tpot.p50` on 32
  reqs divided by the baseline's `1/tpot.p50` on 64 reqs — TPOT is per-token
  so the median should be reasonably stable, but this is a hint, not a
  certified score. Round 2 winner-confirmation runs must use full-N.
