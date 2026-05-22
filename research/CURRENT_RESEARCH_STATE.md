# SENPAI Research State (ib-20260522-r1)

- Date: 2026-05-22 22:55 UTC
- Most recent research direction from human researcher team: none yet (no issues open; repo issues disabled).
- Live blocker: pod did not ship with `pytorch_baseline_metrics.json` or the MMLU-Pro quality-baseline registry. Without them every `evaluate.py` reports `quality_check.pass: false`. r1-tanjiro (PR #4) is generating both before their Sc C arms, using `precompute_quality_baseline.py --backend vllm` (officially supported by `quality_gate.py`) and a vLLM-default reference run for speedup calibration. r1-frieren/r1-fern told to rebase onto the resulting baseline files before their *terminal* full eval and to set `INFERENCE_BENCH_QUALITY_BASELINE_BACKEND=vllm`.
- Current research focus and themes:
  - **InferenceBench**: optimize per-scenario speedup over the PyTorch baseline for
    Mistral-7B-Instruct-v0.3 on a single H100 80GB, 2h budget per run. Paper-facing
    primary metric is scenario-level `speedup_over_pytorch`; quality gate
    (MMLU-Pro 95% of baseline) and integrity gates are hard pre-conditions.
  - **Engine focus**: vLLM 0.11.x for all initial arms. vLLM defaults already give
    the strongest public-reference numbers on C and competitive numbers on A/D;
    HPO-style search tops the leaderboard. SGLang and TGI are fallback options
    once we know how the chosen recipe behaves.
  - **Per-scenario specialization**: A wants short TTFT on long prompts; B wants
    short TPOT on long decodes; C wants high concurrent request throughput;
    D wants balanced p50 metrics under concurrency=4.
  - **Pod topology**: one physical H100 80GB shared by up to 3 logical students
    (r1-frieren, r1-fern, r1-tanjiro). Heavy GPU work must be serialized.
    Lightweight work (launcher prep, smoke tests in a tiny `--quick`, log
    analysis, research notes) can overlap.
- Initial research plan:
  - Round 1, opening salvo (now): one student per scenario among A, B, C. Each
    student owns a single scenario but tries a small, well-justified vLLM-knob
    family before producing one terminal result. Scenario D is left for round 2
    or for a mature recipe to confirm cross-scenario.
  - Each opening hypothesis targets a known high-leverage knob for its scenario:
    - Scenario A (TTFT-heavy, 8192 in): chunked prefill + larger
      `--max-num-batched-tokens` to reduce prefill latency; KV-cache fp8 to free
      memory for prefill batching. Attention backend FLASH_ATTN baseline first.
    - Scenario B (TPOT-heavy, 8192 out): speculative decoding with vLLM n-gram
      method (the search space already supports this) at a small num_speculative_tokens, plus FP8 KV cache to widen decode batch. Decode is the bottleneck; n-gram speculation is high-leverage when prompts/responses share
      n-grams (LongBench-v2 prompts often do).
    - Scenario C (high-load, concurrency=64): large `--max-num-seqs` (256/512) and chunked prefill + prefix caching, plus `--gpu-memory-utilization 0.95`, attention backend FLASH_ATTN. C is throughput, so the goal is to keep the GPU saturated.
- Potential next research directions:
  - Quantized weights (FP8 / AWQ / GPTQ) once we have a clean float baseline; quantization changes both quality risk and latency profile.
  - Attention backend ablation (FLASH_ATTN vs FLASHINFER vs TRITON_ATTN) once the per-scenario sweet spot is known.
  - Cross-scenario confirmation (A-D geomean) for the strongest single-scenario recipes.
  - Scheduler-policy and prefix-caching ablations specifically for C (high-load) — `--enable-prefix-caching` is cheap to try if quality stays.
  - Speculative decoding with longer prompt-lookup windows on B if the initial speculative arm wins.
  - SGLang radix-tree prefix sharing for C (only if vLLM throughput plateaus).
