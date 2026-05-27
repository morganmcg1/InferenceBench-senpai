# SENPAI Inference Baselines — ib-20260527-latest3-r1

- **Advisor branch:** ib-20260527-latest3-r1
- **Tag:** ib-20260527-latest3-r1
- **Target repo:** morganmcg1/InferenceBench-senpai
- **Base model:** mistralai/Mistral-7B-Instruct-v0.3
- **GPU:** RTX PRO 6000 Blackwell ~96GB VRAM (shakedown; not leaderboard-comparable to H100)
- **Scoring seed:** 248 (imported from `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`)
- **W&B project:** `wandb-applied-ai-team/inferencebench-senpai`
- **Time budget:** 2 hour run window
- **Starting point:** vLLM default launcher `src/starting_points/vllm_running/start_server.sh`
- **Preflight:** PASS (require_scoring_preflight.sh, all scenarios, RTX PRO 6000)

## PyTorch Baselines (denominator for speedup)

| Scenario | Workload | Raw objective | PyTorch baseline value | Requests |
|---|---|---|---:|---:|
| A: Input-heavy   | 8192 in / 1024 out, burst c=1   | `1 / ttft.p50` (burst)                                       | 2.281 (ttft.p50=0.4385s) | 128/128 |
| B: Output-heavy  | 1024 in / 8192 out, burst c=1   | `1 / tpot.p50` (burst)                                       | 39.69 (tpot.p50=0.02519s) | 64/64 |
| C: High-load     | 1024 in / 1024 out, 3 profiles  | geomean(`request_throughput_req_per_s`) over burst/poisson/constant | 0.0847 req/s | 768/768 |
| D: General       | 4096 in / 2048 out, burst c=4   | geomean(`1/ttft.p50`, `1/tpot.p50`, `req_thr`) (burst)       | 1.928 | 96/96 |

Speedup is `candidate_raw_objective / pytorch_baseline_raw_objective`. Higher is better. Quality gate is mandatory (`mmlu_pro.ratio >= 0.95`).

## Reference Snapshot (public H100, 2026-05-21)

For direction only. RTX PRO 6000 shakedown numbers are not leaderboard-comparable.

| Method | Sc. A TTFT | Sc. B TPOT | Sc. C req/s | Sc. D geomean | Aggregate |
|---|---:|---:|---:|---:|---:|
| SMAC3 2h vLLM (best public) | 4.37x | 15.23x | 46.70x | 5.69x | 11.53x |
| vLLM default | 1.25x | 2.25x | 48.69x | 1.96x | 4.05x |
| SGLang default | 1.22x | 1.77x | 51.12x | 2.14x | 3.92x |
| PyTorch baseline | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |

## Current Best (terminal, full-eval, validated)

| Scenario | Speedup over PyTorch | Launcher | W&B run | PR |
|---|---:|---|---|---|
| A | — | none | — | — |
| B | — | none | — | — |
| C | **22.48x** | `senpai/launchers/C/sglang-mem085-mrr128/start_server.sh` | `v478wci3` | #124 (merged) |
| D | — | none | — | — |

**C terminal metrics (RTX PRO 6000 seed248):**
- geomean req/s: 1.9041 (PyTorch 0.0847 → 22.48x)
- per profile: burst 2.780 req/s, poisson 2.004 req/s, constant 1.239 req/s
- quality: PASS — mmlu_pro 0.314 vs 0.298 baseline, ratio 1.054, n=500
- VRAM peak: 84,285 MB — fits H100 80GB with margin at 0.85 mem-fraction-static
- 768/768 requests, 0 failures
- engine: SGLang 0.5.9, `--attention-backend triton --mem-fraction-static 0.85 --max-running-requests 128 --chunked-prefill-size 8192 --schedule-policy fcfs`
- validate_result.py: validation_pass=true, baseline_update_allowed=true

## Provisional / Quick / Unconfirmed Candidates

| Scenario | Quick speedup | Arm | PR | W&B | Notes |
|---|---:|---|---|---|---|
| B | **3.44x** | `vllm-ngram-spec` (Arm 1 + ngram speculative `num_spec=5`, `prompt_lookup_min=2`, `max=4`) | #122 frieren | `08pnkhgg` | Quick=4 burst requests; effective gen ~125 tok/s; CUDA graphs ON, prefix caching ON, max-num-seqs=8. Full B (~65 min) unlikely to fit remaining window after tanjiro's full C. |
| C | 4.01x | `sglang-mem085-mrr128` (quick only) | #124 (merged) | `7ccdxfhz` | Quick now superseded by terminal 22.48x full result. |
| B | 1.44x | `vllm-cudagraph-prefix` | #122 frieren | `9w5uw7q3` | Arm 1, baseline. Modest gain over vLLM defaults since CUDA graphs were already on. |
| B | 1.42x | `vllm-cudagraph-block32` | #122 frieren | `3sl0py0b` | Arm 2 flat vs Arm 1; block-size 32 + smaller max-num-batched-tokens not useful at c=1. |
| C | 3.97x | `sglang-default` | #124 tanjiro | `csj0gqi4` | Arm 1, SGLang Triton attention backend, default knobs after libnuma1/libnuma-dev system install. |

## Failed Launches / Dead Ends

(none yet)

## Update History

- 2026-05-27 14:35 — initial ledger created. Preflight PASS for RTX PRO 6000 seed248 scoring assets. Starting search from vLLM default launcher.
- 2026-05-27 15:15 — round 1 quick partials logged. frieren ngram-spec hits 3.44x quick on B (big win). tanjiro SGLang hits 4.01x quick on C (full-eval-bound). fern PR #123 silent since 14:44, second nudge sent.
- 2026-05-27 15:30 — **C WINNER MERGED** PR #124 tanjiro: SGLang sglang-mem085-mrr128 at **22.48x** speedup on C (full eval, quality PASS, validate_result PASS). Tanjiro now idle; new assignment pending. fern woke up, Arm 1 quick A = 1.273x; proceeding to Arm 2. frieren Arm 3 ngram-spec quick 3.44x on B; steering to D pivot.
