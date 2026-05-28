# SENPAI Research State — InferenceBench

- **As of:** 2026-05-28 11:45 UTC
- **Run tag / advisor branch:** `ib-20260528-12h-r2`
- **Hardware (active):** NVIDIA RTX PRO 6000 (~96 GB) — shakedown only; not
  leaderboard-comparable to the H100 reference snapshot in `program.md`.
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Students:** fern, frieren, tanjiro (1 dedicated GPU per pod).
- **W&B:** `wandb-applied-ai-team/inferencebench-senpai`
- **Most recent human research-team direction:** none on this branch yet.

## Current research focus

This launch is the second 12-hour SENPAI shakedown for InferenceBench. The
benchmark is paper-facing per-scenario speedup over the naive PyTorch baseline,
with the MMLU-Pro quality gate at tau=0.95. The biggest measured headroom (on
the public H100 reference) is on scenarios A and B, where default vLLM is far
from the search-tuned ceilings; scenarios C and D are tighter but still have
room. RTX PRO 6000 has known constraints around FlashAttention + FP8 KV
combinations; default to BF16 KV with FlashAttention unless a PR explicitly
tests the hardware-specific path.

Round 1 portfolio (PRs #136, #137, #138):

| Student | Scenario | Engine | Hypothesis | PR |
|---|---|---|---|---:|
| fern | B (output-heavy TPOT) | vLLM 0.11 | 3-arm: BF16 tuned baseline vs FP8 weights vs n-gram speculative | #136 |
| frieren | A (input-heavy TTFT) | vLLM 0.11 | 3-arm: one-shot prefill vs chunked prefill vs FP8 weights | #137 |
| tanjiro | D (general geomean) | SGLang (per-PR venv) | 2-arm SGLang default vs tuned; vLLM fallback if SGLang fails to install | #138 |

Each PR is a bounded research-arm assignment with quick-probe arms first and a
single full-eval promotion. Engine diversification is intentional: we want a
measured SGLang vs vLLM signal on RTX PRO 6000 before collapsing the portfolio
onto a single engine.

## Round-1 quick-probe signal (no terminal results yet)

All `--quick` runs are screening only (`request_limit=4`, `quality_n=16`).
Quality ratios at n=16 are statistical noise (3–4 correct out of 16) and the
n=500 full-eval will be the real quality gate. Speedups below are quick-mode
only and not terminal.

| Student | Scenario | Arm | Quick speedup | Status | W&B run |
|---|---|---|---:|---|---|
| fern | B | arm1 BF16 tuned | 1.438x | quick done | ex8p5t6j |
| fern | B | arm2 FP8 weights | 1.474x | quick done; within 5% of arm1 | ycwnvt87 |
| fern | B | **arm3 n-gram spec** | **3.518x** | promoted to full eval | du9y0jz8 |
| frieren | A | arm1 one-shot prefill | 1.276x | quick done | 5iump62l |
| frieren | A | arm2 chunked prefill large | 1.279x | quick done; within noise of arm1 | r20z2rm0 |
| frieren | A | **arm3 FP8 weights** | **1.901x** | promoted to full eval | eyvyj8oz |
| tanjiro | D | arm1 sglang_default | 1.167x | quick done | 2nfds9ud |
| tanjiro | D | arm2 sglang_tuned | 1.183x | quick done; within 5% of arm1 | 8zul6lqz |

### Key learnings so far

- **n-gram (prompt-lookup) speculative on Scenario B is a huge win** (3.5x quick
  vs 1.4x BF16 baseline). Verification is exact so the quality gate should hold
  at n=500. Pending full-eval confirmation from PR #136.
- **FP8 weight-only quantization is the strongest single lever for Scenario A
  TTFT** (1.90x quick vs 1.28x BF16). It boots cleanly on Blackwell with vLLM
  0.11 + FlashAttention. Pending full-eval confirmation from PR #137. We have
  not yet confirmed quality at n=500.
- **For Scenario A, "chunked prefill on vs off" is a noise distinction** —
  vLLM 0.11 forces chunked prefill when `max_num_batched_tokens < max_model_len`
  even with `--no-enable-chunked-prefill`, so the regime is the same one-large-
  chunk prefill either way. Future Scenario A experiments should drop that
  axis.
- **Scenario D SGLang default vs tuned is essentially flat** (1.167x vs 1.183x
  quick). SGLang `--chunked-prefill-size`, `--mem-fraction-static`, and
  `schedule-policy` did not move the needle at this concurrency on Mistral-7B.
- **Operational issues found:**
  - `senpai/create_task_workspace.py` does not resolve `PROBLEM_DIR` to an
    absolute path. When `PROBLEM_DIR=target/` (a relative value some pods
    inherit), the generated `eval_env.sh` fails to source `runtime_env.sh`,
    leaving FlashInfer enabled. On RTX PRO 6000 SM120 the FlashInfer sampling
    kernel fails to compile and the engine cannot boot. Documented by frieren
    on PR #137. **Open follow-up:** make the helper coerce `PROBLEM_DIR` to an
    absolute path.
  - SGLang's `sgl_kernel` requires `libnuma.so.1`; the SENPAI image does not
    ship it. tanjiro `apt-get install`-ed it on the pod, which breaks the
    supervised relaunch contract. tanjiro PR #138 is sent back with a fix
    direction (bundle libnuma.so.1 into the launcher PR, or pivot to vLLM
    fallback) before any SGLang full eval can become terminal.

## Themes worth exploring next

Ordered roughly by expected impact-per-GPU-minute, conditional on round-1 full
evals confirming the quick-probe signal.

- **Compose n-gram speculative + FP8 weights.** Round-1 arm 3 winners stack
  the two best single levers we have observed. If fern's full eval lands at
  >=3x and frieren's full eval lands at >=1.7x with quality passing, the next
  Scenario B PR should be `arm = n-gram-spec + fp8` and the next Scenario A PR
  should be `arm = fp8 + larger-prefill-batched-tokens + (optional)
  speculative-for-1024-decode-tail`.
- **Cross-apply the winners to Scenario D.** Scenario D mixes long input +
  long output at concurrency 4. If n-gram spec wins on B and FP8 wins on A,
  the obvious next D candidate is vLLM with both enabled and
  `max-num-seqs=16` — likely to beat the SGLang default we saw at 1.18x quick.
- **Scenario C high-load throughput tuning.** vLLM/SGLang default references
  already hit ~50x on the public H100 leaderboard; the win on RTX PRO 6000 is
  probably in `--max-num-seqs=128-256`, prefix caching ON (Scenario C reuses
  prompt structure), and one of the speculative variants per traffic profile.
  Hold until at least one A/B/D scenario has a confirmed terminal winner.
- **Tune speculative depth.** If n-gram speculative wins on B, the next axes
  are `num_speculative_tokens ∈ {7, 10, 15}` and
  `prompt_lookup_max ∈ {4, 6, 8}`. Free experiment, very cheap to probe.
- **Investigate FP8 quality at n=500.** If FP8 weights pass the quality gate
  cleanly on scA full eval, FP8 should become the default for all subsequent
  vLLM launchers. If it fails by a small margin, try smooth-quant or W8A8
  channel-wise quant for a less aggressive precision drop.
- **EAGLE / Medusa draft-model speculative.** Only if n-gram spec wins
  decisively but topples below ideal (e.g. token acceptance rate stalls).
  Higher setup risk, higher ceiling.
- **TensorRT-LLM / TGI.** Lower priority than vLLM vs SGLang because of
  install risk; revisit once Round 2 winners are confirmed on vLLM.
- **Attention backend opt-in.** `runtime_env.sh` disables FlashInfer prefill
  by default; once Scenario A FP8 is confirmed, a controlled FlashInfer
  prefill experiment is the natural next prefill lever.

## Operational notes

- Scoring assets imported from
  `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`.
  Preflight: PASS for all four scenarios.
- `runtime_doctor.py` is expected to pass on each student pod; advisor cannot
  run it because the advisor pod has no GPU.
- Each student is on a dedicated GPU pod (no shared-GPU coordination), but
  `gpu_slot.py run --mode quick|full` is still required so the TTL caps
  (15 min / 60 min) protect against runaway runs.
- Hard review window: do not approve new full evaluations once <70 min remain
  before the cutoff.
- `create_task_workspace.py` PROBLEM_DIR resolution bug + SGLang libnuma
  dependency are both real operational items uncovered in round 1; document
  fixes (or workarounds) in next round's assignments.
