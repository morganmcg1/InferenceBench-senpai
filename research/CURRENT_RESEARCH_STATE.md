# SENPAI Research State

- **Date/time:** 2026-05-24 07:32 UTC (~32 min into the 2 h SENPAI window)
- **Run:** `ib-20260524-ready-r1`, advisor branch `ib-20260524-ready-r1-advisor`
- **Hardware:** NVIDIA RTX PRO 6000 Blackwell (~96 GB), **shakedown evidence only**
- **Model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Most recent human-team directive:** none in GH Issues at boot. (Open Issue #17 is from
  the 2026-05-22 R3 run; not for this advisor branch.)

## Round-1 live state

| PR | Student | Scenario | Status (UTC) |
|---|---|---|---|
| #39 | r1-frieren | C (throughput) | vLLM-FP8 server up (89.8 GiB), `evaluate.py --quick` running in iter 17 (started 07:05:33). Launcher in workspace; not yet pushed to branch. |
| #41 | r1-fern    | B (output-heavy) | Launcher pushed (commits 9a17160→df36f21). Workspace built. Holding for `SLOT-FREE` from PR #39. |
| #42 | r1-tanjiro | A (input-heavy)  | Launcher pushed (commit a9f0de7c). Workspace built, `bash -n` clean, HF cache symlinked. Holding for `SLOT-FREE` from PR #41. |

The pod's Claude watchdog (training-target heuristic that kills any Claude iteration
with no `train.py` process AND a stale log) killed r1-fern's iter 19 once
(code=124 at 07:21:02). An ADVISOR HEARTBEAT comment was posted to all 3 PRs at
~07:25 telling students to commit early/often, take one small step per iteration,
and never assume `/tmp/inferencebench-scenario-*/task/` survives a kill. Since then
the round has stabilized: 4 clean iterations (code=0) and no further watchdog
fires.

## Current Research Focus

Open the first measured RTX PRO 6000 datapoint on each of the four
InferenceBench scenarios under the 2 h SENPAI window. The starting point
(`src/starting_points/vllm_running`) is unmeasured on this hardware; every
scenario's `Current Best Launcher` in `BASELINE.md` is empty. The fastest path
to value is to put one strong-prior recipe on the board per scenario and use
those numbers as the live baselines other PRs must beat.

The reusable building blocks for this hardware are:

- **FP8 weights** (`--quantization fp8`) — native on Blackwell tensor cores, ~7 GB Mistral footprint, halves weight bandwidth per decode step.
- **FP8 KV cache** (`--kv-cache-dtype fp8`) — halves KV bandwidth and KV memory; lets us push concurrency without OOM.
- **`FLASH_ATTN` attention backend** — safer than FlashInfer on Blackwell while we are still in shakedown.
- **vLLM n-gram speculative decoding** — free single-stream decode acceleration when prompts are long and outputs reuse n-grams from the prompt.
- **Chunked prefill off** for single-stream prefill (Scenario A) — chunked prefill helps batched throughput, not single-stream TTFT.
- **Large `max-num-seqs`** for throughput (Scenario C, conc=64) and small `max-num-seqs` for decode-bound (Scenario B, conc=1).

## Themes

1. **One strong vLLM recipe per scenario, then iterate.** Three idle students mapped onto three of the four scenarios (A, B, C). Scenario D (general/balanced) is held for round 2, where it can either confirm a mature winner from A-D or get its own dedicated recipe.
2. **GPU window discipline.** 3 students share 1 GPU pod, so heavy serving runs must be strictly serialized via `SLOT-FREE: ib-20260524-ready-r1` comments. Order: r1-frieren → r1-fern → r1-tanjiro.
3. **Quality gate is a hard pass/fail.** MMLU-Pro tau=0.95 against baseline accuracy ≈ 0.298 (seed 248, 500 samples). FP8 quantization should be safe but is the most likely quality risk.

## Round 1 Assignments

| PR | Student | Scenario | Recipe | GPU window |
|---|---|---|---|---|
| #39 | r1-frieren | C (throughput) | vLLM FP8 weights+KV + chunked prefill + `max-num-seqs 256`, FLASH_ATTN | window 1 |
| #41 | r1-fern    | B (output-heavy) | vLLM FP8 weights+KV + n-gram spec decoding, FLASH_ATTN | window 2 |
| #42 | r1-tanjiro | A (input-heavy) | vLLM FP8 weights+KV + no chunked prefill + `max-num-batched-tokens 16384`, FLASH_ATTN | window 3 |

## Potential Next Research Directions

After round 1 lands, the obvious branches are:

- **Cross-engine sweep.** If vLLM FP8 wins B, A, or D on Blackwell, retry the same logical lever on **SGLang** (`--mem-fraction-static`, `--chunked-prefill-size`, `--schedule-policy lpm`) and **TGI** (`--cuda-graphs`, `--max-batch-prefill-tokens`). H100 SGLang default already wins Scenario C in the public reference — RTX PRO 6000 may invert that.
- **EAGLE/MEDUSA / draft-model speculation** for B if n-gram acceptance is low. EAGLE3 with a tiny TinyLlama-1.1B draft is the obvious upgrade path, but draft loading time eats into the 2 h window — only worth it if n-gram falls short.
- **Compile / CUDA graph tuning** for B and D — `--enforce-eager` vs CUDA-graphs-on, `--compilation-config` levels, FlashInfer prefill backend on Blackwell once we know FA baseline.
- **Higher concurrency for C.** If `max-num-seqs 256` doesn't saturate the GPU, push to 384/512 and re-tune `gpu-memory-utilization`.
- **AWQ/GPTQ** as an alternative quantization for Mistral-7B if FP8 weights fail quality. The pre-quantized AWQ checkpoint of Mistral-7B-Instruct-v0.3 is on Hugging Face.
- **Scenario D recipe.** No round-1 assignment yet; pick after we see which Scenario A/B/C recipe survives, since D borrows from both prefill (A) and decode (B).
- **Cold-start time as a hidden cost.** RTX PRO 6000 + FP8 + vLLM cold launch can be slow; an early-round recipe that prebuilds a per-PR FP8 cache (under `senpai/`, not in protected files) could lift later iterations.

## Plateau / Escalation Plan

If two consecutive PRs on the same scenario fail to beat the live baseline, escalate to:

- A different serving engine (SGLang/TGI/TRT-LLM) for that scenario.
- Lower-precision quantization (FP8 → AWQ INT4 or GPTQ INT4) if quality still passes.
- Speculative decoding for prefill-light/decode-heavy scenarios; remove speculation where it doesn't pay.
- Custom OpenAI-compatible server only as a last resort, since the 2 h window is too tight to write one from scratch.
