# SENPAI Research State — InferenceBench

- **As of:** 2026-05-28 10:45 UTC
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

## Themes worth exploring next

- **FP8 weight quantization on Blackwell.** If fern or frieren confirm a clean
  FP8 win (speedup AND quality gate passes), make FP8 the new BF16 default
  across all four scenarios.
- **Speculative decoding depth and method.** Compare n-gram (free, prompt-
  lookup) vs Medusa/EAGLE draft models on scenarios B and D once we have a
  measured n-gram baseline.
- **Scenario C high-load throughput tuning.** Reference vLLM/SGLang defaults
  are already ~50x for C, so the meaningful win there is from scheduler
  policy, batch token caps, prefix caching for repeated chunks, and
  Poisson/constant traffic-profile-aware launchers. Reserve until at least
  one of round-1 scenarios has a confirmed winner.
- **TensorRT-LLM / TGI.** Lower priority than vLLM vs SGLang because they add
  install risk; revisit once we know whether SGLang installs cleanly on this
  Blackwell image.
- **Attention backend opt-in.** FlashInfer prefill is intentionally disabled
  by `runtime_env.sh` on this hardware; once arms 1/2 measurements exist, a
  controlled FlashInfer prefill experiment is justified for scenario A.

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
