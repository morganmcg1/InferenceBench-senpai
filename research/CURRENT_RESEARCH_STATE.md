# SENPAI Research State — ib-20260524-ready-r3

- **Date:** 2026-05-24 (updated 08:46 UTC)
- **Hardware mode:** RTX PRO 6000 shakedown (not leaderboard-comparable)
- **Time budget:** 2-hour research clock total — expires ~09:05 UTC
- **Advisor branch:** `ib-20260524-ready-r3-advisor`
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`

## Most recent direction from human researcher team

No GitHub Issues from the human researcher team at boot. The operator confirmed
the prepared RTX PRO 6000 scoring assets at
`/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248` pass
hard A–D preflight; this is shakedown evidence, not an H100 leaderboard claim.

## Round 1 status (complete — 2/3 done, 1 in-progress)

- **Slot 1 / PR #37 (r3-frieren, scenario B, FP8 KV):** CLOSED 08:08.
  `speedup_over_pytorch = 1.568x`. **Quality gate FAILED: ratio 0.839 < τ 0.95**.
  Launcher disqualified. W&B: `zm6tpgv4`.

- **Slot 2 / PR #38 (r3-fern, scenario A, FP8 KV):** CLOSED 08:46.
  `speedup_over_pytorch = 1.030x`. **Quality gate FAILED: ratio 0.839 < τ 0.95**.
  Launcher disqualified. Tail-latency win: TTFT p99 = 1.98x. W&B: `b0h44kvt`.
  Key finding confirmed: **FP8 KV is a systematic Mistral-7B quality killer
  (2/2, identical ratio 0.839 — deterministic, not noise).**

- **Slot 3 / PR #40 (r3-tanjiro, scenario D, FP8 KV):** WAITING.
  Slot-3 live since 08:36:45 (fern SLOT-FREE). Advisor wake-up posted 08:43.
  No tanjiro response yet as of 08:46. Hard budget cutoff 09:05 UTC (~19 min).
  Launcher recipe at commit `6762651` is ready. Expect quality FAIL (FP8 KV).
  Still valuable: confirms systematic 3/3 or reveals scenario-D exception.

## Round 2 assignments

- **PR #50 / r3-frieren (scenario B, FP16 KV):** Single lever changed:
  `--kv-cache-dtype auto` vs round-1 `fp8`. Launcher recipe pushed at commit
  `314e8ab`. Slot 4 — queued behind tanjiro. Likely won't run before 09:05
  cutoff; carries to next launch.

- **PR #52 / r3-fern (scenario A, FP16 KV):** Single lever changed:
  `--kv-cache-dtype auto` vs round-1 `fp8`. Slug: `vllm-fp16-bigbatch-prefill`.
  Slot 5 — queued behind frieren. Will carry to next launch.

## BASELINE.md ledger

No SENPAI winners merged yet on any scenario. All round-1 launchers
disqualified by FP8 KV quality gate. Round-2 launchers pending GPU time.

## Key findings so far

1. **FP8 KV cache systematically degrades Mistral-7B-Instruct-v0.3 accuracy
   on RTX PRO 6000 (vLLM).** MMLU-Pro ratio 0.839 on both scenario A and B,
   identical to 3 decimals (16/64 correct at seed 248, n=64). This is real
   hardware/model/engine precision degradation, not per-scenario noise.
   `--kv-cache-dtype auto` (FP16) is the required fix for round 2.

2. **vLLM baseline speed is plausible on RTX PRO 6000.** Scenario B: 1.568x
   over PyTorch. Scenario A: 1.030x (mostly tail win). RTX PRO 6000 PyTorch
   baseline is stronger relative to vLLM than the H100 references suggest —
   potentially due to Blackwell architecture advantages or vLLM FP8 path
   underperformance on Blackwell vs Hopper.

3. **TTFT tail-latency win is real.** Scenario A bigbatch-prefill got 1.98x
   on TTFT p99 despite flat median — the shape of single-batch full-prefill
   is correct, bottleneck is the prefill matmul itself.

4. **senpai/summarize_metrics.py baseline-envelope bug fixed.** Commit
   `3b34a4a`: `--baseline-metrics-json` now unwraps `baseline.profiles.burst`
   correctly. Reported by r3-fern on PR #38.

## Current research focus (round 2)

**Isolate FP8 KV vs FP16 KV across scenarios.** The entire round-2 mandate
is: `--kv-cache-dtype fp8` → `--kv-cache-dtype auto`, identical recipe otherwise.
Expected outcome: quality recovers ≥ 0.95, TPOT/TTFT regresses slightly but
launchers become valid. If FP16 KV also fails, the problem is deeper.

Scenario D (PR #40 / slot 3) still uses FP8 KV in round 1 — expected to
confirm the same quality fail. Round-2 scenario D FP16 variant not yet
assigned (pending tanjiro's round-1 result).

## Potential next research directions (round 3+)

Conditional on round-2 FP16 KV results passing quality:

1. **Full-weight FP8 (`--quantization fp8`) on FP16-KV-passing launcher:**
   Layer another speedup (weight bandwidth reduction) without touching KV
   precision.

2. **n-gram speculative decoding on B / D:** `--speculative-config` with 3–7
   draft tokens to cut TPOT on long-decode workloads.

3. **Attention backend swap:** `VLLM_ATTENTION_BACKEND=FLASHINFER` on
   Blackwell — backend choice may matter more on RTX PRO 6000 than H100
   references imply. Also `TRITON_ATTN`.

4. **`--enable-prefix-caching` probe on scenario A:** evaluate.py chat-template
   wrapping may create shared system-prompt prefix; caching could trim 50–100ms
   off median TTFT.

5. **Block size 32 on scenario B winner:** trade KV fragmentation for fewer
   block-table ops on long-decode.

6. **Scenario C dedicated probe:** `--max-num-seqs 256/512`,
   `--enable-prefix-caching` on poisson/constant profiles where tail-of-request
   reuse exists, `--max-num-batched-tokens` sweep. Low priority round 1 (vLLM
   default already near SMAC3 reference) but could pay off if B/D launchers
   saturate.

7. **SGLang or TGI:** only after vLLM headroom is exhausted on a given
   scenario. Engine swap adds container/runtime debugging cost.

8. **Cross-scenario confirmation run:** only when a single launcher beats
   PyTorch on every scenario — run an A–D sweep and report
   `aggregate/geomean_speedup_over_pytorch`.

## Open risks and watch-items

- **Tanjiro slot 3 at risk:** Pod may be experiencing GitHub API rate-limit
  issues (same as earlier this session). Advisor posted wake-up at 08:43.
  If no response by ~08:55, frieren (PR #50) could take early slot-4 if
  frieren's pod is responsive.
- **Round 2 may not run today:** PR #50 (frieren scB FP16) and PR #52 (fern
  scA FP16) are both queued after tanjiro. Budget expires 09:05. Both carry
  to next launch.
- **Reduced-N comparability:** All speed measurements so far use
  `--request-limit N` (1/8 of full). Valid for round-2 steering, not
  leaderboard claims. Round 2 winner-confirmation runs should use full-N.
- **Quality gate sensitivity:** FP16 KV should restore quality on RTX PRO
  6000 (standard precision), but this hasn't been measured yet on this
  hardware. The round-2 quality eval at n=64 seeds=248 will confirm.
