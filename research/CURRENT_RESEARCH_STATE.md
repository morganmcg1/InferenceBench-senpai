# SENPAI Research State — ib-20260524-leasefix-r1

- **As of:** 2026-05-24 (initial state for this research tag)
- **Most recent human research direction:** No GitHub Issues open. Programme contract remains `target/program.md`: maximise per-scenario `speedup_over_pytorch` on Mistral-7B-Instruct-v0.3 with a 2-hour wall-clock budget per scenario while passing the MMLU-Pro quality gate.

## Current focus

Establish the **first measured SENPAI baseline launchers** on the RTX PRO 6000 Blackwell shakedown hardware. No prior SENPAI measurement exists on this branch, so the immediate priority is to land three terminal-validated launcher recipes that beat the PyTorch baseline on three different scenarios. This both stocks `BASELINE.md` with real numbers and exposes any RTX-PRO-6000-specific surprises in vLLM 0.11.x before we start optimising further.

We have **1 GPU shared by 3 students** for this round, so the three assignments are spread across three different scenarios to minimise serialisation of full evaluations.

## Round 1 assignments

| Student | Scenario | Hypothesis | PR |
|---------|----------|------------|----|
| frieren | A — input-heavy, 1/ttft.p50 | vLLM with chunked prefill, `max_num_batched_tokens=16384`, `FLASH_ATTN`, FP8 KV cache, CUDA graphs, prefix caching | #75 |
| fern    | B — output-heavy, 1/tpot.p50 | vLLM with n-gram speculative decoding (5 tokens, lookup 2–4), FP8 KV cache, CUDA graphs, `FLASH_ATTN` | #76 |
| tanjiro | C — high-load, geomean req/s | vLLM with `max_num_seqs=256`, `max_num_batched_tokens=16384`, chunked prefill, prefix caching, FP8 KV cache, CUDA graphs, `FLASH_ATTN` | #87 |

Common ground: every launcher sources `senpai/runtime_env.sh`, uses `FLASH_ATTN` (FlashInfer remains disabled per the runtime helper), wraps heavy GPU work in `senpai/gpu_slot.py run --wait`, runs quick eval before full, runs a clean relaunch before declaring terminal, and posts a `SENPAI-RESULT` marker per `program.md`.

## Why these three configurations

- **Scenario A (frieren):** TTFT is prefill-dominated. The vLLM 0.11 prefill path benefits most from a batched-token budget that fits the 8K prompt in one or two chunks, plus FP8 KV to clear memory pressure that throttles graph capture. Reference public H100 default vLLM hits only 1.25×; the search-baselines push it to 4.5×. Any clean ≥ 1.5× on RTX PRO 6000 is a first step.
- **Scenario B (fern):** TPOT on a concurrency-1 long-output workload is the cleanest place to play with speculative decoding. n-gram is the safest spec method (exact verification preserves quality), and the SMAC3 H100 result of 15× is largely a speculative-decoding effect. Per-token decode is the metric; CUDA graphs are non-negotiable.
- **Scenario C (tanjiro):** Throughput is batching-dominated. Mistral-7B + 96GB VRAM + FP8 KV cache lets us comfortably push `max_num_seqs` to 256, matching the burst concurrency cap. Public reference is the one scenario where the *default* vLLM is already very strong; modest improvements compound in the aggregate geomean.

## Next research directions (queued for next round)

These are queued as follow-ups based on the round-1 results. Do not assign until round 1 reports.

1. **Scenario D (balanced) launcher** — geomean of TTFT/TPOT/req/s at 4K/2K, burst concurrency 4. Likely a hybrid of the round-1 winners (chunked prefill + moderate `max_num_seqs` + maybe lightweight n-gram spec).
2. **Speculative-decoding sweep on Scenario B** — 3/5/7 tokens, plus draft-model (EAGLE / Medusa / draft-Mistral) variants once the quality gate is known to hold.
3. **Engine bake-off on Scenario C** — SGLang RadixAttention vs vLLM prefix caching. SGLang has known wins on repeated-prefix throughput.
4. **FP8 weights** — if FP8 KV is quality-safe, try FP8 weights too on the throughput scenario; biggest VRAM lever, biggest quality risk. Quality gate is hard.
5. **TRITON_ATTN smoke** — Blackwell sometimes prefers Triton kernels. Run a controlled comparison once a baseline launcher exists per scenario.
6. **Scheduler policy** — vLLM v1 scheduler tweaks (preemption priority, chunked-prefill admission policy); SGLang `--schedule-policy lpm` vs default `fcfs`.
7. **Cross-scenario "universal" launcher** — once we have winners per scenario, run an A–D confirmation on the best general-purpose config and report the aggregate geomean.
8. **Speculative decoding for Scenario D** — D has burst concurrency 4 with 2K outputs, which is in the sweet spot for spec decoding without batch saturation.

## Plateau watch

We are nominally at round 1, so there is no plateau yet. After three rounds of <5% improvement on the same scenario, escalate per the Plateau Protocol in `CLAUDE.md` (change strategy tier; investigate the worst-performing profile; consider non-vLLM engines or kernel-level work).

## Open external context

- The `researcher-agent` was launched in parallel to generate fresh launcher hypotheses and will write to `/research/RESEARCH_IDEAS_2026-05-24_advisor.md`. Round-2 assignments will be informed by that note.
- The headline reference numbers in `program.md` are H100; current results are RTX PRO 6000 shakedown only and **not** leaderboard-comparable.
