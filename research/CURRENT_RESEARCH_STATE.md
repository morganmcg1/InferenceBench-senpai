# SENPAI Research State

- **Date/time:** 2026-05-24 08:04 UTC (~100 min into the 2 h SENPAI window; ~20 min remaining)
- **Run:** `ib-20260524-ready-r1`, advisor branch `ib-20260524-ready-r1-advisor`
- **Hardware:** NVIDIA RTX PRO 6000 Blackwell (~96 GB), **shakedown evidence only**
- **Model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Most recent human-team directive:** none in GH Issues at boot or at 08:04 re-check.

## Live state (round 1 → round 2 in flight)

| PR | Student | Scenario | Status (UTC 08:04) |
|---|---|---|---|
| #39 | r1-frieren | C (throughput) | **MERGED at 07:54Z. 46.35x speedup over PyTorch** (geomean of burst 75.16x / poisson 47.84x / constant 27.69x). Quality PASS (0.9530, margin 0.003 above 0.95 floor). W&B `yiumffcc`. |
| #41 | r1-fern    | B (output-heavy) | vLLM server up, **quick eval 4/4 PASS at 08:03:32Z: TPOT p50 = 3.93 ms ⇒ ~6.4x raw speedup** over PyTorch baseline of 25.15 ms. gen_throughput 210 tok/s. Full eval (64 reqs + 500 MMLU-Pro) running. Three CUDA-header bug-fix commits required (FlashInfer JIT CPATH clash with system CUDA 13). |
| #42 | r1-tanjiro | A (input-heavy)  | Holding for window 3, prep intact (commit `a9f0de7c`). Idle-polling correctly since 07:49. Heartbeat posted at 08:04 with window-3 launch criteria and time-budget warning. |
| #49 | r1-frieren | D (general/balanced) | New assignment opened at 07:58Z. Launcher `senpai/launchers/D/vllm-fp8-balanced/start_server.sh` pushed at 08:01 (`16eff92a`). Recipe matches brief: FP8 weights+KV + chunked prefill + ngram spec + FLASH_ATTN + pre-emptive CPATH fix. Waiting for window 4 (after r1-tanjiro). Time-budget tight; may not fit in remaining 20 min. |

The pod's Claude watchdog killed r1-fern's iter 19 once early in round 1 and again at iter 20
at 07:43:14Z (834 s of idle polling). Both were self-recovered (work pushed before kill).
Mid-round ADVISOR HEARTBEAT (07:25) telling students to commit early/often was effective.

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

## Round 1 / Round 2 Assignments

| PR | Student | Scenario | Recipe | GPU window | Status |
|---|---|---|---|---|---|
| #39 | r1-frieren | C (throughput) | vLLM FP8 weights+KV + chunked prefill + `max-num-seqs 256`, FLASH_ATTN | window 1 | **MERGED 46.35x** ✅ |
| #41 | r1-fern    | B (output-heavy) | vLLM FP8 weights+KV + n-gram spec decoding, FLASH_ATTN | window 2 | quick 6.4x, full eval running |
| #42 | r1-tanjiro | A (input-heavy) | vLLM FP8 weights+KV + no chunked prefill + `max-num-batched-tokens 16384`, FLASH_ATTN | window 3 | holding, prep ready |
| #49 | r1-frieren | D (general/balanced) | vLLM FP8 weights+KV + chunked prefill + ngram spec (n=3) + `max-num-seqs 32`, FLASH_ATTN | window 4 | launcher pushed, holding |

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
