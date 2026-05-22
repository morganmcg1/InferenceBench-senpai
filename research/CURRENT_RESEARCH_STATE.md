# SENPAI Research State

- **Date:** 2026-05-22
- **Launch tag:** `ib-20260522-r5`
- **Advisor branch:** `ib-20260522-r5-advisor`
- **Students this launch:** r5-frieren, r5-fern, r5-tanjiro (3 students packed
  into one pod, 1× H100 80GB total)
- **Wall-clock window:** 2 h per scenario.
- **Most recent human directive:** none received this launch.

## Research focus

Beat the InferenceBench public reference snapshot per-scenario by composing
high-impact serving levers. Public references and gaps to close:

- **Scenario A (TTFT, 8192/1024 burst c=1):** vLLM default 1.25× → search
  ceiling 4.48×. The biggest single-default-to-search delta. Levers: long
  prefill attention (FlashInfer / FA), large `max-num-batched-tokens`, FP8
  KV cache, prefix caching off (varied prompts), CUDA graphs on.
- **Scenario B (TPOT, 1024/8192 burst c=1):** vLLM default 2.25× → SMAC3
  ceiling 15.23×. Biggest absolute ceiling lift. Lever: speculative decoding
  (n-gram lookup or draft), plus FP8 quantization to shrink decode weight I/O.
- **Scenario C (req/s, 1024/1024, 3 traffic profiles):** SGLang default
  51.12× already beats every vLLM search row in the snapshot (search ceiling
  46.70×). Hardest scenario to beat with vLLM-only tuning; the natural attack
  is "SGLang base + targeted tuning" or "vLLM + FP8 + spec dec".
- **Scenario D (geomean of 1/TTFT, 1/TPOT, req/s on 4096/2048 burst c=4):**
  vLLM default 1.96× → SMAC3 ceiling 5.69×. Balanced tuning, geomean of three
  axes. Modest spec decoding plus FP8 KV cache plus chunked prefill is the
  natural starting place.

## Coordination plan

- All three students share a single H100. Heavy GPU work (full evaluator) is
  serialised: each PR's instructions explicitly say "do not run a full
  `evaluate.py` while another r5-* student is on the GPU".
- Quick eval (`evaluate.py --quick`) is cheap; use it to filter launchers
  before the GPU-expensive full eval.
- Skip Scenario C in round 1: highest baseline already, hardest delta.
  Cover A, B, D first; revisit C if any student has spare GPU time.

## Round 1 assignments

- **r5-frieren → Scenario B, n-gram speculative decoding on vLLM (`b-vllm-spec-ngram-fp8`)** —
  biggest absolute ceiling lift. n-gram lookup spec decoding + FP8 weights +
  FlashInfer + FP8 KV cache.
- **r5-fern → Scenario A, vLLM FlashInfer long-prefill recipe
  (`a-vllm-flashinfer-fp8kv`)** — close the 1.25→4.48 gap with FlashInfer +
  large `max-num-batched-tokens` + FP8 KV cache + prefix caching off.
- **r5-tanjiro → Scenario D, vLLM balanced FP8 + light spec decoding
  (`d-vllm-fp8-spec3-flashinfer`)** — balance prefill, decode, throughput.
  Chunked prefill on (concurrency 4), 3-token n-gram spec decoding, FP8 KV
  cache, FlashInfer.

## Potential next research directions

- **Scenario B follow-on:** if n-gram spec decoding lands, try increasing
  `num_speculative_tokens` (3→5→7) and add `prompt_lookup_max=4`; or try a
  Mistral-compatible draft model for higher acceptance.
- **Scenario A follow-on:** if FlashInfer wins, sweep `max-num-batched-tokens`
  in {8192, 12288, 16384} and toggle `enable_chunked_prefill`. Try
  `enforce_eager` off vs. on to isolate CUDA graph cost vs. payoff at c=1.
- **Scenario D follow-on:** widen spec decoding to 5 tokens once acceptance is
  measured; try `attention_backend=FLASHINFER` vs `FLASH_ATTN` head-to-head.
- **Scenario C attack:** SGLang base with `schedule-policy=lpm`,
  `mem-fraction-static=0.90`, `chunked-prefill-size=8192`, then layer FP8 if
  SGLang supports it for Mistral.
- **Cross-scenario confirmation:** once any A/B/D winner lands, run a confirm
  pass on the other scenarios to compute aggregate geomean.
- **Beyond vLLM/SGLang:** TGI with FP8 quantize for B, or TensorRT-LLM build
  if time permits.

## Open risks

- **Spec decoding stability:** n-gram lookup can fail to start on some vLLM
  versions; students must keep a fallback launcher.
- **FP8 quality gate:** the MMLU-Pro τ=0.95 gate can fail with aggressive
  quantization. Students should run `--quick` MMLU before full eval.
- **GPU contention:** three students, one H100. Mis-coordinated full evals
  will corrupt measurements; rely on `kubectl logs` and PR comments to
  serialise.

## Update 2026-05-22 ~22:54 UTC

- **Hardware discrepancy:** the actual pod GPU is **Blackwell sm_120 (~96 GB)**,
  not H100-80 GB as the public reference snapshot assumed. Speedup ratios vs.
  a PyTorch baseline measured on the *same* hardware are still valid; absolute
  speedup numbers are no longer directly comparable to the `program.md`
  leaderboard.
- **vLLM 0.11 + FlashInfer + n-gram spec-dec is broken on this pod.** Asserts
  `decode_wrapper._sm_scale == self.scale` in
  `vllm/v1/attention/backends/flashinfer.py:972`. Discovered by r5-tanjiro on
  PR #13.
  - PR #13 (Scenario D) now pivoted: dropped `--speculative-config`, retrying
    with FP8 + FP8 KV + FlashInfer + chunked prefill only.
  - PR #11 (Scenario B) advised to keep `--speculative-config` and switch
    `VLLM_ATTENTION_BACKEND` → `FLASH_ATTN` instead.
  - PR #12 (Scenario A) unaffected (no spec-dec in that recipe).
- **Reproducible env setup discovered by r5-tanjiro:** local `.vllm_venv` with
  `torch 2.8.0+cu128`, `vllm 0.11.0`, `flashinfer-python 0.6.11.post3`,
  `transformers 4.57`, `setproctitle`, `pyzmq`, plus a `CPATH` snippet
  exposing pip-installed `nvidia/<lib>/include` for FlashInfer's JIT. Other
  students should reuse this rather than reinvent it.

