# SENPAI Research State (ib-20260522-r1)

- Date: 2026-05-23 00:02 UTC (endgame; budget 00:26 UTC, ~24 min left)
- Most recent research direction from human researcher team: none yet (no issues open; repo issues disabled).
- Live environment facts (matter for every assignment in this run):
  - Pod hardware is **RTX PRO 6000 Blackwell (sm_120) ~97 GiB VRAM**, not H100 80GB. Reference-snapshot speedup anchors (1.25x / 2.25x / 48.69x / etc.) are directional on this hardware.
  - vLLM 0.21 on sm_120: flashinfer sampler JIT fails (missing curand headers) — set `VLLM_USE_FLASHINFER_SAMPLER=0`. `VLLM_ATTENTION_BACKEND` env var was dropped — pass `--attention-backend` as a CLI flag instead. CUDA-12 wheel needs `LD_LIBRARY_PATH` for the bundled cu12 runtime (pod has CUDA 13.2).
  - Baseline files **not produced** this round — r1-tanjiro held the GPU 23:01–00:00 UTC running precompute + reference work but never pushed any data files. Their entrypoint last iterated at 22:49 UTC and posted last at 23:26; presumed-stuck Claude session. All Sc A/B/C/D quicks must run with `quality_check.pass=false` labeled, and `--baseline-primary` will fall back to the program.md H100 anchors with explicit caveats.
- Endgame status (2026-05-23 00:02 UTC):
  - r1-tanjiro: PR #4 silent since 23:26, branch tip still `cf13e72`. No baseline data, no Sc C result. Treat as dormant.
  - r1-fern: PR #3 launcher pushed (`f9ef680` speculative_ngram_v1). Approved quick-only terminal. GPU just released 00:00:22 — should grab it now.
  - r1-frieren: PR #2 still on `3602537` (launcher only). Was preempted at 23:56 when tanjiro's second vLLM server reclaimed the GPU. Queued behind fern.
  - Runner.py `_count_chat_tokens` BatchEncoding bug fix landed on advisor as `69b0c3d` (independently identified by frieren + tanjiro).
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
  - **First priority for round 2**: rerun r1-tanjiro's `precompute_quality_baseline.py --backend vllm` + vLLM-default speed reference triplet (Sc A/B/C raw primaries). The team needs these data files committed under `src/eval/inference/baselines/quality/` and `senpai/research/baselines_ib-20260522-r1.json` (or equivalent) so all subsequent quicks/fulls can use `INFERENCE_BENCH_QUALITY_BASELINE_BACKEND=vllm` and compute valid `--baseline-primary`. Budget ~30 min of GPU time for this; do not start hypothesis arms until it lands.
  - **Carry forward all 3 launchers from round 1**: `senpai/launchers/scenario_a/chunked_prefill_v1` (frieren), `senpai/launchers/scenario_b/speculative_ngram_v1` (fern), `senpai/launchers/scenario_c/high_throughput_v1` (tanjiro). These are Blackwell-tested and ready to run if their owner re-grabs the assignment.
  - Quantized weights (FP8 / AWQ / GPTQ) once we have a clean float baseline; quantization changes both quality risk and latency profile.
  - Attention backend ablation (FLASH_ATTN vs FLASHINFER vs TRITON_ATTN) once the per-scenario sweet spot is known. Note flashinfer sampler is broken on sm_120 in this pod, so attention-backend FLASHINFER may also be at risk — confirm with a smoke test before depending on it.
  - Cross-scenario confirmation (A-D geomean) for the strongest single-scenario recipes.
  - Scheduler-policy and prefix-caching ablations specifically for C (high-load) — `--enable-prefix-caching` is cheap to try if quality stays.
  - Speculative decoding with longer prompt-lookup windows on B if the initial speculative arm wins.
  - SGLang radix-tree prefix sharing for C (only if vLLM throughput plateaus).
  - **Operational lessons from round 1**: (a) reserve the first 30 min of a 2h budget for shared tooling (quality baseline, speed references) before any candidate runs; (b) require students to push partial data after each major step, not at session end; (c) student Claude sessions can wedge for 70+ min in a long Bash tool call — consider explicit `timeout` wrappers on heavy GPU-eval commands.
