# SENPAI Research State — InferenceBench

- **Last updated:** 2026-05-23
- **Most recent research direction from human researcher team:** (none yet for
  this launch — boot only)
- **Research tag:** `ib-20260523-rerun-r1`
- **Advisor branch:** `ib-20260523-rerun-r1-advisor`
- **Hardware (shakedown):** 1 x NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM
- **Wall-clock budget:** 2 hours total

## Current research focus and themes

The research goal is to discover inference-serving configurations and systems
recipes for `mistralai/Mistral-7B-Instruct-v0.3` that beat the default vLLM
launcher's PyTorch-relative speedup on each scenario, while preserving the
MMLU-Pro quality gate and integrity rules. All four scenarios are open for
improvement:

- **A: Input-heavy (TTFT p50, burst conc 1)** — prefill latency on 8192-token
  prompts. Levers: chunked prefill size, max-num-batched-tokens, attention
  backend (FlashInfer/Flash/Triton), CUDA graphs, prefix caching off (concurrency
  1 so likely zero hit), eager-vs-compiled.
- **B: Output-heavy (TPOT p50, burst conc 1)** — long decode on 8192-token
  generations. Levers: speculative decoding (n-gram), FP8 KV cache, chunked
  prefill, kv-cache dtype, attention backend, gpu-memory-utilization.
- **C: High-load (req/s geomean across burst, poisson, constant)** — throughput
  on 256 concurrent requests. Levers: max-num-seqs, max-num-batched-tokens,
  prefix caching, kv-cache fp8, scheduler policy, GPU memory utilization.
- **D: General (geomean TTFT/TPOT/throughput, burst conc 4)** — a balanced
  serving profile. Reusable launchers from A/B/C are good candidates.

## Strategy for this 2-hour window

- 1 GPU shared across 3 students. Serialize heavy benchmark runs to keep
  measurements clean. Students working off the GPU prepare launchers, study
  metrics, or analyze prior runs.
- Each student owns a single scenario for their first PR, focused on the
  highest-leverage knobs. Coordinated GPU slot order in PR bodies.
- All terminal results must come from a clean relaunch + full `evaluate.py`
  run, not from the live training shell.

## Round 1 assignments (boot)

- **r1-frieren → Scenario C:** vLLM with aggressive batching (`max_num_seqs`,
  `max_num_batched_tokens`), `enable_prefix_caching`, `kv_cache_dtype=fp8`,
  high `gpu_memory_utilization`. Highest speedup headroom (H100 ref 46.70x).
- **r1-fern → Scenario B:** vLLM with n-gram speculative decoding,
  `kv_cache_dtype=fp8`, CUDA graphs (no `--enforce-eager`), high gpu mem util.
  Long decode benefits the most from token-level acceleration.
- **r1-tanjiro → Scenario A:** vLLM with chunked prefill at a high
  `max_num_batched_tokens`, FlashInfer attention backend,
  `kv_cache_dtype=fp8`, `--enforce-eager=false`. Long prefill is compute-bound.

## Potential next research directions

- **SGLang as alternative engine** on whichever scenario shows the weakest vLLM
  result (especially Scenario A/D where SGLang's RadixAttention can help, but
  burst concurrency 1 may erase that).
- **TGI** as a fallback engine for stability.
- **Speculative decoding variants:** longer `num_speculative_tokens` for
  Scenario B; EAGLE / Medusa if vLLM supports a path from CLI.
- **Quantization sweep:** GPTQ / AWQ / FP8 weights — only if quality gate still
  passes.
- **Attention backend ablation:** FLASH_ATTN vs FLASHINFER vs TRITON_ATTN per
  scenario.
- **Scheduler policy ablation:** `--scheduling-policy` priority vs fcfs for
  Scenario C poisson/constant traffic.
- **Cross-scenario confirmation runs** for mature winners — once a launcher
  beats baseline on one scenario, sanity-check A-D for the geomean leaderboard.
- **Compile/graph capture caveats** on RTX PRO 6000 Blackwell — measure
  warm-up cost before declaring CUDA graph wins.
