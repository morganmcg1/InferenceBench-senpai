# SENPAI Research State — InferenceBench R3

- Date: 2026-05-22
- Advisor branch: `ib-20260522-r3-advisor`
- Time budget: 2 hours total
- Hardware: 1x H100 80GB shared across 3 packed students (r3-frieren, r3-fern, r3-tanjiro)
- Base model: `mistralai/Mistral-7B-Instruct-v0.3`
- W&B project: `wandb-applied-ai-team/inferencebench-senpai`

## Latest direction from human research team

None received this launch. Default to the program contract.

## Current research focus

Beat the public reference snapshot (SMAC3 11.53x aggregate) on each scenario by
hand-designing **single-shot strong launchers**. With only a 2h total budget and
3 students sharing 1 GPU, we cannot reproduce SMAC3 from scratch — instead we
take its near-final configurations as a strong starting point and push beyond
them with levers SMAC3's flat search space cannot easily reach:

1. **FP8 model weight quantization + FP8 KV cache.** Mistral-7B has a known-good
   FP8 path in vLLM; ~2x memory headroom → larger effective batch/concurrency
   on every scenario.
2. **N-gram speculative decoding** for output-heavy / decode-bound scenarios
   (B and D). Long decode windows mean N-gram prompt-lookup wins.
3. **Aggressive batching and prefix caching for the high-load scenario** (C),
   where vLLM default already beats PyTorch 48.7x but SMAC3 only reaches 46.7x —
   the *default* vLLM is already near the high-load ceiling, and most gains come
   from removing tail-latency-only tuning. Stay close to default-but-tuned.
4. **Chunked prefill + FlashInfer for input-heavy** (A), which is the lowest
   speedup scenario (4.37x SMAC3) and where prefill scheduling matters most.

We are running one scenario per student in this first round.

## Round 1 assignments

| Student | Scenario | Hypothesis slug | Headline lever |
|---|---|---|---|
| r3-tanjiro | A (input heavy) | vllm-fp8-flashinfer-chunked-prefill-A | FP8 + FlashInfer + chunked prefill, optimize TTFT |
| r3-frieren | B (output heavy) | vllm-fp8-ngram-spec-B | FP8 + KV FP8 + N-gram speculative decoding, optimize TPOT |
| r3-fern | C (high load) | vllm-fp8-large-batch-C | FP8 + KV FP8 + 256 seqs + 16384 batched tokens, optimize req/s |

GPU coordination order (PLANNED): r3-frieren → r3-fern → r3-tanjiro.

GPU coordination (ACTUAL, 22:48 UTC): r3-fern actually took the GPU first
(observed ~22:47 by r3-tanjiro). r3-tanjiro pushed their launcher and is
waiting politely. r3-frieren has not yet pushed code or started — they have
been nudged.

## Compatibility notes discovered live (Round 1 hardware setup)

- **vLLM 0.21.0 + FlashInfer 0.6.8.post1** is the installed stack.
- **`VLLM_ATTENTION_BACKEND` env var is unrecognized in vLLM 0.21** — use
  CLI flag `--attention-backend FLASHINFER` (or `FLASH_ATTN`) instead.
  vLLM 0.21 auto-selects FLASHINFER for fp8 KV on Blackwell anyway
  (r3-frieren confirmed at runtime: `Using FLASHINFER attention backend out of
  potential backends: ['FLASHINFER', 'TRITON_ATTN']`).
- **FlashInfer JIT compile needs `curand.h`**, missing initially from
  `/usr/local/cuda/include/`. r3-tanjiro fixed it by symlinking from
  `/usr/local/lib/python3.10/dist-packages/nvidia/curand/include/curand.h`
  into `/usr/local/cuda/include/curand.h`. This unblocks all FP8/FlashInfer
  attempts. Both r3-fern's first attempt (22:46) and r3-tanjiro's first
  attempt failed on this before the fix.
- **FlashInfer FP8 JIT additionally needs `LD_LIBRARY_PATH` and `CPATH`
  workarounds** — r3-frieren documented these in PR #5 commit `6b027ee`.
  `LD_LIBRARY_PATH` must include the nvidia pip-installed lib paths
  (`/usr/local/lib/python3.10/dist-packages/nvidia/{cuda_nvrtc,cuda_runtime,...}/lib`)
  for `libnvJitLink.so.12` resolution. `CPATH` must include the FlashInfer
  FP8 JIT include path.
- **Hardware appears to be sm_120f, not sm_90 (H100)** — total GPU memory
  ≈ 97887 MiB suggests Blackwell-class H100 96GB or B100. The reference
  snapshot in `program.md` is for H100 80GB / Mistral-7B / 2h. Our results
  may not be directly comparable; report our measured speedup anyway and
  let the human team triangulate.
- **CUDA graph capture with FP8 + FlashInfer crashes silently** on this
  hardware during r3-tanjiro's run — switching to `FLASH_ATTN` backend and
  reducing `--gpu-memory-utilization` from 0.95 → 0.90 worked as fallback.
  Further fallback if needed: `--enforce-eager` (disables cudagraphs).
- **vLLM emits suboptimality warning** when `max_num_batched_tokens` is too
  tight relative to speculative-decoding draft count: `max_num_scheduled_tokens
  is set to 4096 based on the speculative decoding settings. Consider increasing
  max_num_batched_tokens to accommodate the additional draft token slots.`
  Discovered by r3-frieren on PR #5 — v2 launcher bumps to 8192.

## R3 launch infrastructure blocker (filed as issue #17)

**Quality gate and `scenario/<X>/speedup_over_pytorch` cannot be computed**
because two baseline files are absent from this harness checkout:

1. `src/eval/inference/baselines/quality/mistralai_Mistral-7B-Instruct-v0.3_torch.json`
   — `runner.py` short-circuits with `error:"missing baseline accuracy for: mmlu_pro"`
   when missing, so MMLU-Pro inference is never run.
2. `pytorch_baseline_metrics.json` per-scenario — needed by
   `senpai/log_metrics_to_wandb.py` for the paper-facing speedup ratio.

Impact: r3-frieren's PR #5 has clean recipe (TPOT p50 = 5.09 ms, 64/64
success on burst, 93GB/97GB VRAM) but cannot prove > 15.23x SMAC3 reference.
Rough position by manual estimation is ~7-16x, bracketing SMAC3.

Mitigation for remaining R3 students: report **raw primary metrics** in
SENPAI-RESULT markers (e.g., `scenario/<X>/inverse_<metric>_p50`). Ignore
the quality gate short-circuit error. Do not attempt to edit benchmark
files. Wait for human team to land the baseline files before any merge.

## Round 1 status (live, 23:55 UTC)

| Student | PR | Scenario | Status | Result |
|---|---|---|---|---|
| r3-frieren | #5 | B (TPOT) | held WIP, queued v2 | TPOT p50 = 5.09 ms, 196.4 tok/s, 64/64. Recipe works, awaits baseline for speedup. |
| r3-fern | #6 | C (req/s) | WIP, claiming GPU | (eval pending, ETA ~30 min) |
| r3-tanjiro | #10 | A (TTFT) | WIP, waiting GPU | (likely quick-eval-only tail slot ~00:25 UTC) |

## Plausible next-round directions

If we beat the SMAC3 reference on one or more scenarios:
- Confirm with a cross-scenario A-D full eval on the winning launcher.
- Push speculative decoding to higher n_tokens (5→7) on B.
- Try TensorRT-LLM or SGLang fp8 as an alternative-engine challenger.

If we miss the SMAC3 reference:
- Drop FP8 model weights but keep FP8 KV cache (quality concerns).
- Restrict to "vLLM default + 1 lever per PR" to isolate the failed lever.
- Try AWQ/GPTQ pre-quantized Mistral checkpoints if they pass quality.

## Notes

- vLLM is the only engine where SMAC3/TPE measurably beats defaults at this
  scale, so default engine choice is vLLM unless a hypothesis explicitly
  benchmarks SGLang/TGI/TensorRT-LLM.
- Quality gate (MMLU-Pro ratio >= 0.95 of PyTorch) is the binding integrity
  constraint for any quantization choice. Always validate via full evaluator.
- Final winner must launch in foreground via `exec ...` in
  `senpai/launchers/<scenario>/<slug>/start_server.sh` and survive a clean
  supervised relaunch.
