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

## Update 2026-05-22 ~23:40 UTC

- **FlashInfer + sm_120 root cause confirmed:** the assertion fires from
  FlashInfer kernel warmup (`kernel_warmup.py:55 _dummy_run(num_tokens=16,
  create_mixed_batch=True)`), *independent* of speculative decoding. The
  TRTLLM bypass `supports_trtllm_attention()` in `vllm/utils/flashinfer.py:191`
  only returns True on **SM 100** (Hopper), so sm_120 (Blackwell RTX PRO 6000)
  cannot escape the regular decode-wrapper path. Confirmed by both r5-tanjiro
  (PR #13) and r5-frieren (PR #11).
- **All three students now on Triton/FA-2-3 attention:** PR #11, #12, #13 all
  using `VLLM_ATTENTION_BACKEND=FLASH_ATTN` or default Triton on sm_120; no
  FlashInfer in any terminal arm this launch.
- **r5-fern's FlashInfer venv-patch experiment (smoke-only, not for terminal):**
  Two patches made FlashInfer functional on a tiny smoke prompt:
    1. Relax `decode_wrapper._sm_scale == self.scale` assert (line ~972) to
       `math.isclose` + overwrite — the strict equality check is overly
       conservative; the computed scale matches numerically.
    2. Probe `_cached_module.plan()` arity in `fast_plan_decode` (line ~1146):
       vLLM 0.11.0 hardcodes a 15-arg call but FlashInfer 0.5.3+ extended the
       ABI to 19 args (`window_left, fixed_split_size=-1, disable_split_kv=False,
       num_colocated_ctas=0`); try-except appends extras on TypeError.
  These patches live in `.vllm_venv` and do not survive a clean supervised
  relaunch, and the 10-token smoke does not validate 8K-prefill paths.
  **Useful follow-up:** worth filing upstream against vLLM 0.11 / FlashInfer
  0.6.11 ABI mismatch, and worth re-attempting a Scenario A FlashInfer run
  once those land in a pinned wheel.
- **`--tokenizer-mode mistral` SPM bug (r5-frieren):** rejects every
  `/v1/{chat/,}completions` on this SPM Mistral with `Expected
  special_token_policy to be a SpecialTokenPolicy, got <class 'NoneType'>`
  (vLLM 0.11.0 sets `_special_token_policy=None` for SPM; only Tekken gets a
  default). Use default `auto` mode — returns `LlamaTokenizerFast`, decodes
  correctly.
- **Baseline strategy this launch:** no precomputed PyTorch baseline exists
  on the Blackwell pod. r5-tanjiro (PR #13) is producing the Scenario D
  PyTorch baseline via `precompute_baseline.py` + `transformers_openai_server`
  on this hardware (output: `senpai/research/baselines/pytorch_baseline_D_blackwell.json`).
  PR #11 and #12 will report raw `1/tpot.p50` and `1/ttft.p50` as
  `primary_metric.value` sentinels and label `Baseline mode:
  raw_primary_objective` clearly in their bodies; the
  `speedup_over_pytorch` ratios will be computed post-hoc by the advisor once
  baselines exist. **No fabricated H100→Blackwell extrapolations** allowed —
  the relative PyTorch/vLLM gap is hardware-dependent.
- **GPU queue at 23:48 UTC:** r5-frieren running quick eval on Scenario B
  (PR #11, GPU 87933 MiB @ 94%); r5-fern queued (PR #12, launcher already
  updated to FLASH_ATTN); r5-tanjiro queued (PR #13, will run Scenario D
  PyTorch baseline first, then optional FA + FP8 + FP8 KV candidate if
  >20 min remains in wall-clock).

