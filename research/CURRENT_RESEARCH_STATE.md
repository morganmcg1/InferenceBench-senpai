# SENPAI Research State — InferenceBench

- **As of:** 2026-05-28 14:15 UTC
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

Round 1 portfolio — status (PRs #136–#138), Round 2 open (#139, #140):

| Student | Scenario | Engine | Hypothesis | PR | Status |
|---|---|---|---|---:|---|
| fern | B (output-heavy TPOT) | vLLM 0.11 | n-gram speculative (quick 3.52x → full 2.687x) | #136 | **MERGED 12:19 UTC — Sc B best 2.687x** |
| fern | C (high-load throughput) | vLLM 0.11 | high-conc BF16 + prefix caching (arm1 BF16 high-conc, full 21.052x) | #140 | **MERGED 13:03 UTC — Sc C first winner 21.052x** |
| fern | C (high-load throughput) | vLLM 0.11 | scale to max-num-seqs 128/256, batched-tokens 16384 | #142 | **MERGED 13:44 UTC — Sc C new best 21.098x** |
| fern | C (high-load throughput) | SGLang | SGLang default + chunked prefill + radix cache vs vLLM | #144 | **assigned 13:50 UTC** |
| frieren | A (input-heavy TTFT) | vLLM 0.11 | FP8 weights (quick 1.90x → full 1.87x) | #137 | **MERGED — Sc A best 1.866x** |
| tanjiro | D (general geomean) | SGLang per-PR venv | sglang_default + bundled libnuma (full 1.2506x terminal) | #138 | **MERGED 12:50 UTC — Sc D first winner 1.247x** |
| tanjiro | B (output-heavy TPOT) | vLLM 0.11 | n-gram spec depth sweep (tokens 7/10/15, lookup 4/6/8) | #141 | **MERGED 14:10 UTC — Sc B new best 3.550x (+32% over PR #136)** |
| tanjiro | B (output-heavy TPOT) | vLLM 0.11 | FP8 weights + spec15/lookup8 composition (extend PR #141) | #146 | **assigned 14:15 UTC** |
| frieren | D (general geomean) | vLLM 0.11 | FP8 + n-gram composition (arm3 full 2.073x) | #139 | **MERGED 13:19 UTC — Sc D new best 2.073x (supersedes #138)** |
| frieren | A (input-heavy TTFT) | vLLM 0.11 | FP8 + n-gram spec on Sc A (extend #137 winner) | #143 | **CLOSED 14:00 UTC — full eval 1.8603x (-0.30% noise), spec metric-orthogonal on TTFT-only Sc A** |
| frieren | A (input-heavy TTFT) | vLLM 0.11 | FP8 weights + FP8 KV cache to attack prefill attention BW | #145 | **assigned 14:00 UTC** |

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
| fern | B | **arm3 n-gram spec** | **3.518x** | full eval 2.687x merged | du9y0jz8 |
| frieren | A | arm1 one-shot prefill | 1.276x | quick done | 5iump62l |
| frieren | A | arm2 chunked prefill large | 1.279x | quick done; within noise of arm1 | r20z2rm0 |
| frieren | A | **arm3 FP8 weights** | **1.901x** | full eval 1.866x merged | eyvyj8oz |
| tanjiro | D | arm1 sglang_default | 1.167x | full eval 1.2506x (relaunch-safe, in review) | 2nfds9ud → nf10i0y2 |
| tanjiro | D | arm2 sglang_tuned | 1.183x | dropped per simplicity tiebreak | 8zul6lqz |
| fern | C | arm1 BF16 high-conc | **3.888x** | full eval **21.052x merged** | qvcrnc4s → tebmnnza |
| fern | C | arm2 + prefix cache | 3.928x | within 1% of arm1; arm1 preferred | 2rtv6gxb |
| fern | C | arm3 + FP8 | 3.798x | FP8 hurts on high-conc throughput (-3.3%) | wuvd1ycq |
| frieren | D | arm1 fp8 only | 1.326x | quick done | (no W&B) |
| frieren | D | arm2 ngram only | 1.697x | quick done | (no W&B) |
| frieren | D | **arm3 fp8+ngram** | **1.810x** | promoted to full eval (running) | (no W&B yet) |

### Key learnings so far

- **n-gram (prompt-lookup) speculative on Scenario B is a confirmed win** (3.5x quick
  → full eval **2.687x speedup_over_pytorch**, quality 0.300/0.298 = 1.007 ratio,
  n=500, 64/64 speed success, W&B `dav3txgq`; **PR #136 merged 12:19 UTC**).
  Quick-to-full drop (3.5x → 2.7x) reflects lower speculative acceptance rate over
  the full 64-request Sc B distribution (longer outputs diverge from prefix context).
  Sc B now beats H100 vLLM default (2.25x); ceiling from SMAC3 at 15x suggests
  deeper speculative depth and multi-seq batching have large untapped headroom.
- **FP8 + n-gram speculative compose on Scenario D** (frieren quick: arm1 FP8-only
  1.326x, arm2 ngram-only 1.697x, arm3 composition **1.810x** — +6.6% over arm2,
  +36.5% over arm1). The interaction is constructive: FP8 cuts the 4096-token
  prefill TTFT while n-gram speculation cuts the 2048-token decode TPOT, and the
  two operate at different stages of each request. Full eval in flight to confirm.
- **Scenario C high-concurrency exploration (fern):** arm1 BF16 high-conc with
  `--max-num-seqs 64 --max-num-batched-tokens 8192` returns **3.888x quick speedup**
  (geomean req/s) — well above Sc A/B quick levels, consistent with H100 reference
  pattern where vLLM default already hits 48.69x on Sc C. Prefix caching is +1%
  noise at quick n=4 (low inter-request prefix overlap on Sc C's varied prompts).
  **FP8 hurts on Sc C** (-3.3% vs arm2): high-concurrency throughput is bound by
  KV-cache memory and scheduler overhead, not weight-read bandwidth, so FP8
  dequant overhead on every fwd pass dominates the marginal bandwidth savings.
  Different optimum than Sc A. Arm1 (simplest, within 1% of best) promoted to
  full eval.
- **Tanjiro Sc D SGLang relaunch-safe** (1.2506x full eval, MMLU-Pro 0.31 vs
  baseline 0.298 = ratio 1.04, 96/96 speed success, W&B `nf10i0y2`). The
  bundled-libnuma fix (Path A) succeeded — `libnuma.so.1` shipped under
  `senpai/launchers/D/tanjiro-sglang/lib/` + `LD_LIBRARY_PATH` prepend +
  `start_server.sh` auto-bootstraps the SGLang venv via `uv venv` if missing.
  PR #138 currently `CONFLICTING` on `start_server.sh` (PR #136 committed a
  different version at merge); rebase send-back at 12:44 UTC.
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

## Round-2 quick-probe signals (in flight)

| Student | Scenario | Arm | Quick speedup | Status | W&B run |
|---|---|---|---:|---|---|
| fern | C | arm1 seqs128/tok8192 | 3.9017x | full eval **21.098x** (rebase blocked, +0.22% vs #140) | 2jw0s5lk |
| fern | C | arm2 seqs64/tok16384 | 3.8991x | within 0.1% of arm1 | (logged) |
| fern | C | arm3 seqs128/tok16384 | 3.9005x | within 0.04% of arm1 | (logged) |
| fern | C | arm4 seqs256/tok16384 | 3.8704x | -0.8% vs arm1 | (logged) |
| tanjiro | B | arm1 spec7/lookup4 | 2.121x | quick done | cwbiqekn |
| tanjiro | B | arm2 spec10/lookup6 | 2.857x | quick done; +35% vs arm1 | ahowy8xt |
| tanjiro | B | **arm3 spec15/lookup8** | **6.918x** | promoted to full eval (VRAM 90.16 GB) | rdmg3q7u |
| frieren | A | arm1 FP8+spec5/lookup4 | 1.9115x | promoted to full eval (simpler within 5%) | quc7zeij |
| frieren | A | arm2 FP8+spec10/lookup6 | 1.9143x | +0.14% vs arm1 — wins TPOT but Sc A is TTFT only | 3gp3w4s5 |

### Round-2 key learnings

- **Sc C concurrency scaling has hit a plateau.** All 4 fern arms within 0.8%
  at quick — burst concurrency=64 fits in every `max-num-seqs >= 64` config.
  Only the full eval (256 reqs/profile) shows the +0.22% from doubling seqs
  to 128 — at noise floor. **Implication:** further vLLM concurrency tuning
  on Sc C is exhausted; next Sc C lever is engine (SGLang) or model-side
  (larger batches via dtype FP8 KV-cache, prefix caching for poisson profile).
- **N-gram speculative depth on Sc B scales aggressively.** arm1 (depth 7) →
  arm2 (depth 10) → arm3 (depth 15) returned 2.121x → 2.857x → 6.918x quick —
  superlinear. Sc B's 8192-token outputs give long horizons for n-gram
  match-and-accept. Even with full-eval haircut (PR #136 saw 3.5x quick →
  2.687x full, ~24% drop), arm3 should land ~5.2x — a **2x improvement** over
  the current Sc B baseline.
- **Sc A speculative composition is prefill-limited.** arm1 (spec5) and arm2
  (spec10) tie at TTFT (1.91x both) but diverge on TPOT (-17% in arm2's
  favor). Sc A primary metric is `1/ttft.p50`, so the deeper window doesn't
  help. Frieren correctly promoted arm1 (simpler within 5%). Composition
  still beats FP8-only quick (1.901x → 1.911x — +0.5% from n-gram on the
  1024-token decode tail), so the FP8+n-gram pattern is genuinely
  cross-scenario and not just a Sc D coincidence.

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
