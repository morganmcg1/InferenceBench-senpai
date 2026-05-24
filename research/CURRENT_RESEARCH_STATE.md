# SENPAI Research State — ib-20260524-ready-r5

- **Last updated:** 2026-05-24 08:50Z
- **Most recent human directive:** (none on this branch; using `program.md`
  research contract and the operator-set 2-hour shakedown clock)
- **Active hardware:** RTX PRO 6000 Blackwell (96 GB), shakedown only
- **Active model:** Mistral-7B-Instruct-v0.3

## Round 1 results

| Scenario | PR | Speedup | Status |
|---|---|---|---|
| B (output-heavy / TPOT) | #34 (r5-frieren) | **2.40×** | **MERGED** — new best |
| A (input-heavy / TTFT) | #35 (r5-fern) | TBD | WIP — awaiting SLOT-FREE from r5-frieren |
| D (general / balanced) | #36 (r5-tanjiro) | TBD | WIP — awaiting SLOT-FREE from r5-fern |

## Active work

- **r5-frieren** — Assigned PR #53 (Scenario B round 2: n-gram 7 tokens ablation). First action: post SLOT-FREE on PR #35 and tear down server. Runs GPU last in the queue.
- **r5-fern** — PR #35 (Scenario A: FP8 + max-num-batched-tokens 16384). Waiting for SLOT-FREE from r5-frieren.
- **r5-tanjiro** — PR #36 (Scenario D: FP8 + FP8 KV + chunked prefill + prefix caching + ngram spec 3 tokens). Waiting for SLOT-FREE from r5-fern.

## Shared-GPU coordination contract

- One heavy full eval at a time on the shared pod. Students post `SLOT-FREE` in their PR comments when their server is fully torn down.
- Sequence for this round: r5-frieren SLOT-FREE → r5-fern Scenario A → r5-fern SLOT-FREE → r5-tanjiro Scenario D → r5-tanjiro SLOT-FREE → r5-frieren Scenario B round 2 (if time permits).
- Time budget: ~1 hour remaining in 2h shakedown window (as of 08:45Z).

## Key findings so far (RTX PRO 6000 shakedown)

**Scenario B:** FP8 weights + FP8 KV cache + n-gram speculation (5 tokens) → **2.40×**
- TPOT p50: 25.15 ms → 10.47 ms (2.4×); generation throughput: 39 → 97 tok/s
- MMLU-Pro: 0.286 (0.960 × baseline, pass)
- Blackwell sm_120f requires: `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_DISABLE_FLASHINFER_PREFILL=1`, `VLLM_ATTENTION_BACKEND=TRITON_ATTN`
- Bug fixed: `senpai/summarize_metrics.py::_load_metrics` PyTorch baseline JSON unwrap (now on branch)

## Potential next research directions

- **Scenario B round 2 (assigned):** n-gram 7 tokens (PR #53, r5-frieren) — single A/B vs merged 2.40× winner
- **Scenario B round 3:** Calibrated FP8 KV scaling for Mistral-7B-Instruct-v0.3 — closes quality gap, enables more aggressive quantization
- **Scenario B round 4:** EAGLE-3 or MLP-speculator to push past the n-gram acceptance ceiling (~2.5×)
- **Scenario A (in progress):** FP8 + large max-num-batched-tokens, target TTFT improvement
- **Scenario D (in progress):** FP8 + FP8 KV + chunked prefill + prefix caching + ngram spec
- **Scenario C (deferred):** SGLang with radix prefix caching; benefits from round-1 characterization first
- **Cross-scenario confirmation:** After A/D results, aggregate geomean speedup across A–D
