# SENPAI BASELINE — Scenario B (Output-Heavy)

- Branch: `ib-20260528-scen-b-r1`
- Launched: 2026-05-28
- Setting: RTX PRO 6000 Blackwell shakedown (NOT leaderboard-comparable to H100)
- Base model: `mistralai/Mistral-7B-Instruct-v0.3`
- Scenario: B (input ~1024 / output ~8192, 64 requests, burst, concurrency 1)
- Primary metric: `scenario/B/speedup_over_pytorch` = `(1/candidate_tpot_p50_burst) / (1/baseline_tpot_p50_burst)`
- Quality gate: MMLU-Pro 500-question subset, tau=0.95 of PyTorch baseline accuracy 0.298

## PyTorch reference baseline (must beat this to score speedup>1)
- File: `src/eval/inference/baselines/speed/torch/inference_scenario_b_output_heavy/mistralai_Mistral-7B-Instruct-v0.3/baseline_metrics.json`
- `requests_sha256`: `6a7bb79e0cde7cf24334c6b60da319c7bf458c756e75067f28c1770e0f4b869d`
- Burst TPOT p50 = **0.02515 s/token** → raw objective `1/tpot.p50` = **39.76 tok/s**
- TTFT p50 = 0.0709 s; ITL p50 = 0.0146 s; gen throughput = 39.19 tok/s
- 64/64 successful requests, failure rate 0.0

## Public reference (2026-05-21 H100, 2h budget) — NOT directly comparable to RTX PRO 6000
| Method | Sc. B TPOT speedup |
|---|---:|
| SMAC3 search, 2h vLLM | 15.23x |
| TPE search, 2h vLLM | 14.76x |
| Random search, 2h vLLM | 11.34x |
| Best listed agent (Sonnet 4.6) | 12.03x |
| vLLM default, no agent | 2.25x |
| SGLang default, no agent | 1.77x |
| HF TGI default, no agent | 1.37x |
| PyTorch baseline | 1.00x |

Reference H100 numbers above are search direction only. RTX PRO 6000 Blackwell
results can differ; need on-hardware measurement.

## Current best terminal launcher (this branch)

### 2026-05-28 18:22 — PR #159: Scenario B n-gram speculative decoding (frieren F1) — TERMINAL WINNER

- **Primary metric:** `scenario/B/speedup_over_pytorch` = **2.8115x**
- **TPOT p50 (full, 64 requests):** 8.94 ms/tok (raw `inverse_tpot_p50` = 111.78 tok/s)
- **Quality (MMLU-Pro n=500):** 0.314 (ratio 1.054 vs PyTorch baseline 0.298, τ=0.95 → pass)
- **Speed:** 64/64 requests successful, failure_rate = 0.0
- **Validation:** `validation_pass=true`, `baseline_update_allowed=true`, `terminal_eligible=true`
- **W&B run:** `6r6w0ukx`
- **Launcher:** `senpai/launchers/scenario_b/f1_ngram_k5/start_server.sh`
- **Reproduce:**
  ```bash
  cd "target/"
  # vLLM + n-gram speculative decoding, k=5, BF16, CUDA graphs on
  python -m vllm.entrypoints.openai.api_server \
    --model "mistralai/Mistral-7B-Instruct-v0.3" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --disable-log-stats \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'
  ```

> Note: quick screening showed F1 at 3.50x (TPOT 7.18 ms at n=4 burst), while the full 64-request burst
> settled at 2.81x (TPOT 8.94 ms). Acceptance dynamics at full concurrency 1 with long outputs are
> slightly less favourable than short-burst probes but still a strong win.

## Evidence ledger

### Terminal result

| Source | Arm | TPOT p50 (full) | Speedup vs PT | MMLU-Pro (n=500) | Quality ratio | W&B |
|---|---|---:|---:|---:|---:|---|
| PR #159 frieren | **F1 BF16 + ngram k=5 (lookup_max=4, min=2)** | **8.94 ms** | **2.81x** | **0.314** | **1.054** | `6r6w0ukx` |

### Quick / screening ledger (all `result_kind=research_signal`, NOT terminal)

| Source | Arm | TPOT p50 (quick) | Speedup vs PT | MMLU-Pro screening (n=16) | W&B |
|---|---|---:|---:|---|---|
| PR #159 frieren | F0 vLLM default | ~17.5 ms | 1.44x | 0.250 / 0.298 = 0.84 | o39u89pa |
| PR #159 frieren | F1 BF16 + ngram k=5 (lookup_max=4, min=2) | 7.18 ms | 3.50x | 0.250 / 0.298 = 0.84 | 3rvxd2tx |
| PR #159 frieren | F2 BF16 + ngram k=3 | 7.67 ms | 3.28x | 0.188 / 0.298 = 0.63 | smaj0ntn |
| PR #159 frieren | F3 FP8 weights + ngram k=5 | 7.29 ms | 3.45x | (screening pass) | yittyrbl |
| PR #159 frieren | F4 BF16 + ngram spec=25 lookup=12 | ~4.5 ms | 5.60x | 0.188 / 0.298 = 0.63 | s5aj7628 |
| PR #159 frieren | F5 BF16 + ngram spec=25 lookup=15 | ~6.0 ms | 4.19x | 0.188 / 0.298 = 0.63 | oaveolmf |
| PR #162 fern    | N3 FP8 weights + ngram k=3 + lean sched | 10.38 ms | 2.42x | 0.313 / 0.298 = 1.05 | vpymvy8e |

> Quick screening overstated F1 speed vs full eval (3.50x at n=4 burst → 2.81x at n=64 burst).
> N-gram acceptance dynamics differ at full concurrency 1 with long `ignore_eos=true` outputs.

## Failed / negative findings
- **FP8 weight quantization adds dequant overhead** at decode on this RTX
  PRO 6000 / vLLM 0.11 build: F3 (FP8+k=5) = 3.45x vs F1 (BF16+k=5) =
  3.50x (~1% slower); N3 (FP8+k=3+lean) = 2.42x vs F2 (BF16+k=3) = 3.28x
  (~26% slower per TPOT). Two independent measurements agree.
- Aggressive ngram windows (spec=25) trade quality for speed (n=16 ratio
  0.63 is below the vLLM-default screening noise floor of 0.84).

## Round-1 lessons (apply to next round)
1. **Full Scenario B eval is ~57-60 min on this hardware** with the F1
   launcher; quick exploration must finish by T+~30 min so a full eval
   starts no later than T+~60 min and completes by T+~120 min.
2. **The "quick→full" promotion rule needs a hard advisor gate** — once
   the best quick arm is identified, the advisor should freeze further
   exploration on that PR and force the full eval slot. Round-1 quick
   exploration continued well past F1's clear lead.
3. **Quick screening at n=16 is too small** to discriminate quality. A
   medium screening (n=64-100 MMLU-Pro) on the promoted candidate would
   give a much stronger pre-full signal.
4. **Push launchers before each evaluation slot** — both students were
   slow to push and frieren never pushed her launcher recipes inside the
   2h window. Mandate `git push` as part of every quick-eval flow.

## Update history
- 2026-05-28 16:40 UTC — Branch opened; preflight passes via
  `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`;
  PyTorch baseline metrics available; quality registry pinned.
- 2026-05-28 17:25 UTC — Frieren posted 3 quick evals (F0/F1/F2);
  F1 leading at 3.50x. Fern silent.
- 2026-05-28 17:30 UTC — Frieren added F3 (FP8+k=5 = 3.45x), correctly
  rejected as not stacking.
- 2026-05-28 17:40 UTC — Fern posted N3 (FP8+k=3+lean = 2.42x). FP8
  finding cross-confirmed.
- 2026-05-28 17:39-17:44 UTC — Frieren explored aggressive spec windows
  (F4/F5 spec=25, 5.60x / 4.19x with quality degradation).
- 2026-05-28 18:17 UTC — Advisor sent urgent stop-and-push message.
- 2026-05-28 18:22 UTC — **Frieren posted terminal SENPAI-RESULT**: F1 full eval
  complete. Speedup 2.81x, TPOT p50 8.94 ms, quality 0.314 (ratio 1.054, n=500),
  validation passed. PR #159 merged by advisor at 18:25 UTC. **This is the
  round-1 terminal baseline: 2.81x.**
