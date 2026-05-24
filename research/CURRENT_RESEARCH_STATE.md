# SENPAI Research State — ib-20260524-ready-r2

- **As of:** 2026-05-24 08:02 UTC (T+62 min of 120 min program)
- **Most recent direction from the human research team:** No new issues. Operator note at boot — RTX PRO 6000 scoring assets at `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`. Shakedown run, NOT leaderboard claim.
- **Mode:** RTX PRO 6000 shakedown — Blackwell ~96GB, 1 shared GPU, 2-hour total program.
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`.

## First round status

| Slot | Student | Scenario | Status | Result |
|---|---|---|---|---|
| 1 | r2-frieren | B (output-heavy, TPOT) | ✓ DONE — MERGED | **2.91x** speedup / quality 0.3203 (PASS) — PR #43 |
| 2 | r2-fern | A (input-heavy, TTFT) | SLOT-FREE received, server starting | — |
| 3 | r2-tanjiro | D (general, balanced) | Waiting for Fern SLOT-FREE | — |

**GPU:** Released by Frieren at 07:56:26 UTC. Fern (slot 2) was notified at 07:58; her slot cap is 08:32.
**Time remaining:** ~58 min (ending ~09:00 UTC).

## Hardware discoveries this round (critical for all students)

1. **FlashInfer JIT is broken** on this pod — CUDA 13 nvcc vs FlashInfer 0.6.11 bundled CUDA-12 headers. All FlashInfer paths fail at JIT compile time. Use `VLLM_ATTENTION_BACKEND=FLASH_ATTN` unconditionally.
2. **FP8 KV cache is inactive** — same nvcc issue. `--kv-cache-dtype fp8` gets wired via env var but falls back to bf16 at runtime. Both launchers (Fern, Tanjiro) should expect bf16 KV; FP8 headroom not available this session.
3. **CUDA graphs DO work** — Frieren confirmed clean capture (~17s cold relaunch with cached graphs). Keep `--enforce-eager` absent.

## Current best valid launchers

| Scenario | Speedup | Launcher | PR | W&B |
|---|---:|---|---|---|
| B: Output-heavy (TPOT) | **2.91x** | `senpai/launchers/B/ngram-spec-fp8-kv/start_server.sh` | #43 | fgzox7fi |
| A: Input-heavy (TTFT) | — (in progress) | — | #44 | — |
| D: General (geomean) | — (queued) | — | #45 | — |
| C: High-load (req/s) | — (deferred) | — | — | — |

## What round 1 is testing (unchanged)

1. **FP8 KV cache** — wired in all launchers, currently inactive due to pod nvcc issue.
2. **CUDA graphs** — confirmed working.
3. **n-gram speculative decoding** — Frieren used `{"method":"ngram","num_speculative_tokens":5}`, Tanjiro uses `num_speculative_tokens":3` (smaller, appropriate for concurrency-4).
4. **Attention backend variation** — Frieren: FLASH_ATTN (FlashInfer broken). Fern: FLASH_ATTN (same). Tanjiro: FLASH_ATTN (same).

## Fern's adjusted config for slot 2 (scenario A)

Original plan was FLASHINFER backend — skip it. Use `VLLM_ATTENTION_BACKEND=FLASH_ATTN` directly:
```bash
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" --host "${HOST}" --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" --gpu-memory-utilization 0.92 \
  --kv-cache-dtype fp8 \   # will fall back to bf16 — still correct flags
  --max-num-seqs 16 --max-num-batched-tokens 16384 \
  --no-enable-chunked-prefill --no-enable-prefix-caching \
  --trust-remote-code --disable-log-stats
```
No speculative decoding for A (short 1024-token output, prefill-bound — spec decoding doesn't help).

## Potential next research directions (round 2)

1. **EAGLE draft model for scenario B** — Mistral-7B EAGLE weights exist on Hub. This is the path to 10x+ TPOT; prompt_lookup caps at ~3x.
2. **num_speculative_tokens=3 variant for B** — cheap, expected +5-10% TPOT.
3. **FP8 resolution** — if nvcc can be upgraded to match FlashInfer 0.6.11 CUDA-12 headers, FP8 KV activates and frees ~10-15 GB.
4. **Scenario C arm** — "default + CUDA graphs" may still improve on 48.69x if CUDA-graph overhead was a factor in the original vLLM-default measurement.
5. **Full n=64 eval for scenario B** — confirms the n=8 estimate.
6. **Compound wins** — merge all clean round-1 knobs into BASELINE for a combined A+B+D re-run.
