# SENPAI Research Results — `ib-20260528-scen-c-r1`

## 2026-05-28 18:22 — PR #175 (CLOSED, exploratory): vLLM k=7 n-gram quick probe

- **Branch**: `scen-c-frieren/vllm-ngram-k7-quick-probe`
- **Hypothesis**: `num_speculative_tokens=7` may amortize TPOT further than k=5 in Scenario C's long-decode regime if draft-token acceptance is high enough that verifier-rejection cost doesn't dominate; alternatively, rejection overhead may neutralize or reverse the gain.
- **Eval mode**: quick (4 reqs/profile, 12 reqs total). **Not promoted to full eval — no time budget, no baseline-update intent.**

### Result

| Metric | k=5 quick (PR #165 Arm E) | k=7 quick (this PR) | Delta |
|---|---|---|---|
| Quick speedup_over_pytorch | ~4.07x | **4.24x** | +0.17x (+4%) |
| Quick geomean req/s | (not in PR #165 body) | 0.3595 req/s | — |
| Speed requests | 12/12 | 12/12 | identical |
| MMLU-Pro n=16 quick screen | failed (noise) | failed (3/16 ratio 0.629) | same regime |
| VRAM peak | — | 89.29 GB | — |
| W&B run | (PR #165) | 05zrttlc | — |

### Analysis and conclusions

- k=7 quick beats k=5 quick by ~4% — a small bump consistent with k=7 accepting 1-2 additional draft tokens per decode round without the verifier-rejection cost washing out the gain.
- The MMLU-Pro n=16 quick screen failed (3/16 correct vs expected ~4.77/16), but the same noise affected the k=5 quick screen — PR #165's full n=500 eval passed quality at ratio 1.000, so there's no reason to think k=7 would be qualitatively worse (verifier scoring is unchanged by k).
- **Quick mode is screening only**. The 4 reqs/profile sample is dominated by per-request overhead and is not representative of the 768-request full-eval steady state. The +4% quick edge could be amplified, neutralized, or reversed at full eval.

### Next directions implied by this result (for future Scenario C launches)

1. **Paired k=5 vs k=7 full evaluations** (192 req/profile each, n=500 quality) — required to know if the +4% quick edge survives the long-decode regime.
2. k=9 worth probing **only if k=7 full beats k=5 full**. Diminishing returns are expected per acceptance probability decay.
3. **Operational hazard logged**: SGLang server orphan on port 8000 — see CURRENT_RESEARCH_STATE.md research findings for mitigation suggestions.

## 2026-05-28 17:54 — PR #165: vLLM Scenario C — n-gram speculative decoding wins

- **Branch**: `scen-c-frieren/vllm-c-chunked-prefill-full-speculative`
- **Hypothesis (Arm D)**: chunked-prefill OFF wins over chunked-prefill ON in full eval, reversing the quick-mode noise.
- **Hypothesis (Arm E)**: n-gram speculative decoding amortizes TPOT in Scenario C's long-decode regime (output_len=1024, ignore_eos=true). The Mistral-Instruct chat structure produces enough repeated token spans for prompt-lookup speculation to hit.

### Diagnostic finding (Arm D)

`--no-enable-chunked-prefill` is a **no-op in vLLM 0.11.0 V1**: `arg_utils.py:1548` unconditionally sets `enable_chunked_prefill=True` for non-pooling tasks. The PR #161 quick-mode A-vs-D delta was therefore run-to-run noise. The student diagnosed this from `server.log` (final scheduler config printed) and pivoted to Arm E. Important repo-wide note for future scenarios.

### Terminal result (Arm E full eval)

| Metric | Value |
|---|---|
| scenario/C/speedup_over_pytorch | **23.98x** (vs prior baseline 22.18x → +8.1%) |
| geomean req/s | 2.0313 req/s |
| quality/mmlu_pro_observed_accuracy | 0.298 (ratio **1.000**, PASS at n=500) |
| speed requests | 768/768 succeeded |
| VRAM peak | ~89 GB / 96 GB |
| W&B run | mj8f07f0 |
| validation_pass | true |

### Per-profile metrics

| Profile | req/s | gen tok/s | TTFT p50 | TPOT p50 |
|---|---:|---:|---:|---:|
| burst (c=64) | 2.947 | 30.01 | 0.096s | 0.0331s |
| poisson (r=32, c=32) | 2.109 | 39.72 | 0.077s | 0.0253s |
| constant (r=16, c=16) | 1.349 | 49.84 | 0.070s | 0.0205s |

### Analysis and conclusions

- vLLM + Arm A config + n-gram spec decode beats both pure-vLLM (20.84x) and SGLang (22.18x). Per-profile burst req/s 2.95 vs 2.74 for SGLang shows the speculation gain comes from decode-amortization, not from a per-profile concurrency advantage.
- Quality ratio exactly 1.000 — n-gram speculation is exact (proposed tokens are verified against the model's own logits), so accuracy is preserved.
- Per-profile TPOT improved (burst 0.033s vs PR #161 Arm A would have been higher) — spec decode is doing real work; the burst profile gets the largest gain in raw req/s.
- Self-contained launcher (no env-var dependency, in contrast to PR #163 SGLang launcher).

### Next directions implied by this result

1. Test higher `num_speculative_tokens` (7, 9) — diminishing returns expected but unverified.
2. Combine n-gram spec decode with SGLang's faster scheduler — would require SGLang spec-decode support (EAGLE/MEDUSA) and a draft model.
3. Explore vLLM `--speculative-config` with `method=eagle` if a Mistral-7B EAGLE checkpoint is available.

## 2026-05-28 17:31 — PR #163: SGLang Scenario C boot + tune defaults

- **Branch**: `scen-c-fern/sglang-c-throughput-tune`
- **Hypothesis**: Paper reference shows SGLang default at 51.12x on H100 — highest of all engines for Scenario C. If RTX PRO 6000 SM120 can boot SGLang with `--attention-backend triton` (the only RTX-safe option in the search space) and pass MMLU-Pro quality, even a fraction of the reference 51.12x would be a strong Scenario C baseline. Tuning `max_running_requests`, `mem_fraction_static`, and `schedule-policy` probes additional headroom.

### Setup notes (RTX PRO 6000 SM120 specific)

- SGLang 0.5.9 installed in per-PR venv via `uv venv` + `uv pip install sglang[all]`. `create_engine_venv.py` failed because the stock Python had no `ensurepip`.
- Required apt package: `libnuma1` for `sgl_kernel` to load.
- `--sampling-backend pytorch` required (no FlashInfer on SM120).
- `--attention-backend triton` required (FlashAttention/FlashInfer not used).

### Quick arm results

| Arm | mem_frac | max_running | sched | chunked_prefill | Quick speedup | W&B |
|---|---:|---:|---|---:|---:|---|
| A0 | 0.85 | default | fcfs | 8192 (default) | 3.88x | okdhvh3h |
| A | 0.88 | 256 | fcfs | 4096 | **3.96x** ← promoted | uadpj28k |
| B | 0.88 | 256 | lpm | 4096 | 3.92x | (file wiped) |

LPM vs FCFS scheduling: essentially tied; FCFS marginally ahead. Scenario C requests have no meaningful prefix overlap so RadixAttention/LPM doesn't help.

### Terminal result (Arm A full eval)

| Metric | Value |
|---|---|
| scenario/C/speedup_over_pytorch | **22.18x** |
| geomean req/s | 1.879 req/s |
| quality/mmlu_pro_observed_accuracy | 0.314 (ratio 1.054, PASS) |
| speed requests | 768/768 succeeded |
| VRAM peak | 86.7 GB / 96 GB |
| W&B run | b4xhsfby |
| validation_pass | true |

### Per-profile metrics

| Profile | req/s | gen tok/s | TTFT p50 | TPOT p50 |
|---|---:|---:|---:|---:|
| burst (c=64) | 2.739 | 25.82 | 0.089s | 0.037s |
| poisson (r=32, c=32) | 1.983 | 36.37 | 0.043s | 0.026s |
| constant (r=16, c=16) | 1.221 | 43.90 | 0.039s | 0.022s |

### Analysis and conclusions

- SGLang beats vLLM Arm A (20.84x) by +6.4% on the same RTX PRO 6000 shakedown hardware. Burst-profile req/s 2.74 vs vLLM Arm A's burst is the dominant gap, suggesting SGLang's continuous-batching + RadixAttention layout is better at the wave-of-c=64 pattern even when prefix caching is irrelevant.
- VRAM headroom: 9.3 GB of 96 GB unused. There is room to push `mem_fraction_static` higher (0.90-0.92) and `max_running_requests` up (384-512).
- FCFS ≥ LPM here (Scenario C prompts are diverse).
- Quality is *better* than torch baseline (ratio 1.054), so any speed-favoring tweak should still aim for ratio ≥ 0.95.

### Next directions implied by this result

1. **Push SGLang concurrency**: max_running_requests 384/512, mem_fraction_static 0.90-0.92, chunked-prefill-size 8192 — try to extract the remaining VRAM headroom.
2. **Hardcode the winning config as launcher defaults** so reproduce doesn't depend on caller env vars.
3. **Test SGLang `--enable-torch-compile`** (search space says false, but worth a quick probe).
4. Frieren's PR #165 (Arm D vLLM, chunked-prefill OFF) — still pending; if it wins the steady-state hypothesis, even-money for the lead.

## 2026-05-28 17:19 — PR #161: vLLM Scenario C high-concurrency throughput sweep

- **Branch**: `scen-c-frieren/vllm-c-throughput-sweep`
- **Hypothesis**: Pushing vLLM's `max_num_seqs`, `max_num_batched_tokens`, and `gpu_memory_utilization` above the conservative starting-point defaults improves Scenario C geomean throughput. RTX PRO 6000 has ~96GB VRAM vs H100's 80GB, giving headroom. Chunked prefill on + BF16 KV (FP8 KV incompatible with FlashAttention on this hardware).

### Quick arm results

| Arm | max_num_seqs | max_num_batched_tokens | gpu_mem_util | chunked_prefill | Quick geomean speedup | W&B |
|---|---:|---:|---:|---|---:|---|
| A | 384 | 16384 | 0.92 | ON | 2.79x (screening) | j5er5g2l |
| B | 512 | 16384 | 0.93 | ON | 2.76x (screening) | 0szs1hnx |
| D | 384 | 16384 | 0.92 | OFF | 3.80x (screening) | 8ipbzguo |

### Terminal result (Arm A full eval)

| Metric | Value |
|---|---|
| scenario/C/speedup_over_pytorch | **20.84x** |
| geomean req/s | 1.765 req/s |
| PyTorch baseline geomean req/s | 0.0847 req/s |
| quality/mmlu_pro_observed_accuracy | 0.306 (ratio 1.027, PASS) |
| speed requests | 768/768 succeeded |
| W&B run | btpqa2rl |
| validation_pass | true |
| baseline_update_allowed | true |

### Analysis and conclusions

- **Arm A (chunked prefill ON) chosen for full eval** because the quick-mode Arm D advantage (+37% vs Arm A quick) is a small-N artifact: all burst requests arrive concurrently in quick mode (4 req), so chunked prefill just serializes what would otherwise be one batch. Under full eval (256 req/burst, c=64 = 4 waves), chunked prefill should help by overlapping wave-2 prefill with wave-1 decode.
- **Arm B ≈ Arm A** in quick mode — extra max_num_seqs headroom isn't exercised at 4 req/profile.
- VRAM peak stable at 89-91% with gpu_mem_util=0.92.
- **Open question**: Does Arm D (chunked-prefill OFF) beat Arm A in full eval? Needs a full eval run to confirm. If the steady-state argument is right, Arm A should win; if Arm D wins, it implies prefill scheduling is not the bottleneck for 4-wave burst.

### Next directions implied by this result

1. Full eval of Arm D (chunked-prefill OFF) to resolve the open question — high priority.
2. N-gram speculative decoding for the long-decode phase of Scenario C (output_len=1024, ignore_eos=true).
3. SGLang default (PR #163 / fern) booted and passed quality at A0; its full eval is now the most important pending measurement.
