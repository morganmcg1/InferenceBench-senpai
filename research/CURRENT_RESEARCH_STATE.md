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

## Compatibility notes discovered live

- vLLM 0.21.0 + FlashInfer 0.6.8.post1 is the installed stack.
- `VLLM_ATTENTION_BACKEND` env var is unrecognized in vLLM 0.21 — students
  must use the CLI flag `--attention-backend FLASHINFER` instead. I have
  patched the original PR bodies via comments for r3-frieren and r3-fern.

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
