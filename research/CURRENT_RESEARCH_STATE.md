# SENPAI Research State — ib-20260524-ready-r2

- **As of:** 2026-05-24 08:39 UTC (T+99 min of 120 min program)
- **Most recent direction from the human research team:** No new issues. Operator note at boot — RTX PRO 6000 scoring assets at `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`. Shakedown run, NOT leaderboard claim.
- **Mode:** RTX PRO 6000 shakedown — Blackwell ~96GB, 1 shared GPU, 2-hour total program.
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`.

## First round status

| Slot | Student | Scenario | Status | Result |
|---|---|---|---|---|
| 1 | r2-frieren | B (output-heavy, TPOT) | ✓ DONE — MERGED | **2.91x** speedup / quality 0.3203 (PASS) — PR #43 |
| 2 | r2-fern | A (input-heavy, TTFT) | ✓ DONE — MERGED (negative baseline) | **1.236x** (floor — FlashInfer + FP8 KV both inactive) — PR #44 |
| 3 | r2-tanjiro | D (general, balanced) | RUNNING since 08:38 UTC — ~21 min remain | — |

**GPU:** Tanjiro loaded GPU at ~08:38 UTC. Slot cap: ~09:00 UTC (program end).
**Time remaining:** ~21 min (ending ~09:00 UTC).

## Hardware discoveries this round (critical for all students)

1. **FlashInfer JIT is broken** on this pod — CUDA 13 nvcc vs FlashInfer 0.6.11 bundled CUDA-12 headers. All FlashInfer paths fail at JIT compile time. Use `VLLM_ATTENTION_BACKEND=FLASH_ATTN` unconditionally.
2. **FP8 KV cache is inactive** — same nvcc issue. `--kv-cache-dtype fp8` gets wired via env var but falls back to bf16 at runtime. Expect bf16 KV; FP8 headroom not available this session.
3. **CUDA graphs DO work** — Frieren confirmed clean capture (~17s cold relaunch with cached graphs). Keep `--enforce-eager` absent.
4. **vLLM v0.11.0 ignores `--no-enable-chunked-prefill`** when `--max-num-batched-tokens >= max-model-len` — scheduler forces chunked prefill regardless. Not necessarily harmful; just be aware.

## Current best valid launchers

| Scenario | Speedup | Launcher | PR | W&B |
|---|---:|---|---|---|
| A: Input-heavy (TTFT) | **1.236x** | `senpai/launchers/A/flashinfer-fp8-kv/start_server.sh` | #44 | 6h1b2dk7 |
| B: Output-heavy (TPOT) | **2.91x** | `senpai/launchers/B/ngram-spec-fp8-kv/start_server.sh` | #43 | fgzox7fi |
| D: General (geomean) | — (in progress) | — | #45 | — |
| C: High-load (req/s) | — (deferred) | — | — | — |

## What round 1 tested

1. **FP8 KV cache** — wired in all launchers, currently inactive due to pod nvcc issue. ~10-15 GB headroom deferred.
2. **CUDA graphs** — confirmed working. Contributed estimated 1.3–1.5x factor to B's 2.91x.
3. **n-gram speculative decoding** — Frieren used `num_speculative_tokens=5` → 2.91x on B. Tanjiro uses `num_speculative_tokens=3` at concurrency 4.
4. **Attention backend** — FLASH_ATTN used universally (FlashInfer broken).
5. **Chunked prefill** — Tanjiro testing for scenario D concurrency-4 balanced workload.

## Potential next research directions (round 2)

**For scenario B (highest headroom — current 2.91x vs H100 ceiling 15.23x):**
1. **EAGLE draft model** — Mistral-7B EAGLE weights on Hub. This is the path to 10x+ TPOT; prompt_lookup caps at ~3x. Highest-priority round-2 target.
2. **num_speculative_tokens=3 variant** — cheap, expected +5-10% TPOT over current 5-token config.
3. **Increase prompt_lookup_max 4→6-8** for longer lookups on high-repetition outputs.

**For scenario A (currently at floor — 1.236x vs H100 ceiling 4.37x):**
1. **FlashInfer 0.7.x nightly** — CUDA-13 headers shipped ~2026-04. `pip install flashinfer-python==0.7.0` could unlock FlashInfer attention kernel AND FP8 KV simultaneously. Highest-leverage single fix for A.
2. **xFormers or TRITON_ATTN_VLLM_V1** — no JIT dependency; may give better long-prefill perf.
3. **Prefix caching** — scenario A has 8192-token inputs; shared prefix could cut TTFT dramatically.
4. **Fix quality-gate baseline registry** — eval pipeline `create_task_workspace.py` looks for `unknown_model_torch.json`; separate bug-fix PR.

**For scenario D (Tanjiro currently running):**
- Result pending. Chunked prefill + n-gram spec + CUDA graphs. Stop rule: <1.5x → drop spec → if still <1.5x post negative.

**Cross-cutting:**
1. **FP8 KV resolution** — upgrade nvcc to CUDA-12 compat or install FlashInfer 0.7.x nightly. Unlocks ~10-15 GB VRAM, enabling larger batches.
2. **Full n=64 eval for scenario B** — confirms the n=8 estimate at higher statistical confidence.
3. **Compound wins** — after round 1 knobs confirmed, run combined A+B+D with all working optimizations merged.
