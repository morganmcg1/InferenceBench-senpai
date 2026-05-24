# SENPAI Research State

- **Date/time:** 2026-05-24 ~17:30 UTC (round 2 of advisor invocations)
- **Human research team directives:** None received. Infrastructure hotfixes applied by morganmcg1 to live pods.

## Critical Infrastructure Constraints (MUST READ for all assignments)

Discovered and confirmed during round 1. Every launcher recipe MUST obey:
- ❌ NO `VLLM_ATTENTION_BACKEND=FLASHINFER` — FlashInfer JIT fails on RTX PRO 6000 sm_120
- ❌ NO `--kv-cache-dtype fp8` with FlashAttention backend
- ✅ `--kv-cache-dtype fp8` works with TRITON_ATTN (no explicit backend env var needed)
- ✅ `source senpai/runtime_env.sh` sets VLLM_USE_FLASHINFER_SAMPLER=0, VLLM_DISABLE_FLASHINFER_PREFILL=1, INFERENCE_BENCH_MAX_MODEL_LEN=32768
- ✅ n-gram speculative decoding works (pure Python config, unrelated to FlashInfer)
- ❌ NO weight FP8 quantization (`--quantization fp8`) — quality gate margin too thin (ratio=0.953 on Sc D)

## Current Research Focus

Baseline measurement phase on RTX PRO 6000. One scenario measured (D).
Key question now: can we get clean measurements across A, B, C before time expires?

**Active assignments:**

| Student | PR | Scenario | Status |
|---|---|---|---|
| r2-frieren | #54 | B (output-heavy, TPOT) | WIP — waiting for GPU slot; updated launcher guidance posted |
| r2-fern | #58 | A (input-heavy, TTFT) | WIP — holds GPU slot ~17:29+; eval in progress |
| r2-tanjiro | #71 | C (high-load throughput) | NEW — prepare launcher now, takes slot after r2-fern + r2-frieren |

**Completed:**

| Student | PR | Scenario | Result |
|---|---|---|---|
| r2-tanjiro | #64 (merged) | D (balanced) | 1.284× speedup, quality=0.953 PASS |

## Per-Scenario State

| Scenario | Best speedup | Recipe | Status |
|---|---:|---|---|
| A | unmeasured | — | r2-fern running |
| B | unmeasured | — | r2-frieren waiting (updated launcher ready) |
| C | unmeasured | — | r2-tanjiro assigned (PR #71) |
| D | **1.284×** | TRITON_ATTN + FP8 KV + chunked prefill | merged |

## Proven Working Recipe (use as template for all new launchers)

```bash
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" --host "${HOST}" --port "${PORT}" \
    --max-model-len "${INFERENCE_BENCH_MAX_MODEL_LEN:-32768}" \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code --disable-log-stats \
    --kv-cache-dtype fp8 \
    --enable-chunked-prefill \
    [scenario-specific: --max-num-batched-tokens X --max-num-seqs Y]
```
- Do NOT set `VLLM_ATTENTION_BACKEND=FLASHINFER`
- Always `source senpai/runtime_env.sh` first

## Potential Next Research Directions (post round 2)

1. **Sc D: prefix caching follow-up** — single-lever `--enable-prefix-caching` on proven recipe
2. **Sc D: n-gram speculative decoding** — add 5-token n-gram to 2K output at concurrency=4
3. **Sc B: n-gram speculative decoding** (primary hypothesis still untested) — if r2-frieren succeeds
4. **Cross-scenario confirmation** — if A/B/C all have results, run A-D sweep to get aggregate geomean
5. **SGLang exploration** — if vLLM plateau, try SGLang with TRITON attention (different scheduler, may handle high-concurrency differently)
6. **Larger max-model-len** — if any scenario benefits from 64K context (would need VLLM_ALLOW_LONG_MAX_MODEL_LEN=1)
