# SENPAI Research State — InferenceBench (ib-20260522-r2)

- Current date and time: 2026-05-22
- Most recent research direction from human researcher team: none received yet
- Advisor branch: `ib-20260522-r2-advisor`
- W&B project: `wandb-applied-ai-team/inferencebench-senpai`
- Active students: r2-frieren, r2-fern, r2-tanjiro (3 logical students, 1 shared H100 80GB)

## Current research focus and themes

The 2-hour InferenceBench window is split across the four serving scenarios.
The reference snapshot (`target/program.md`, 2026-05-21) shows SMAC3-tuned vLLM
already reaching strong numbers (A 4.37x, B 15.23x, C 46.70x, D 5.69x), so the
research goal is to push past those references with hand-designed launchers
that combine known high-leverage knobs.

Round 1 (dispatched 2026-05-22) covers three different scenarios so the
launchers do not collide and learnings can compound:

- **Scenario A — long-prefill TTFT** (r2-fern): TTFT is dominated by prefill
  kernel choice, chunked prefill behavior, and KV-cache memory headroom.
  Hypothesis: vLLM + FP8 weights + FP8 KV cache + FlashInfer attention + larger
  `--max-num-batched-tokens`, with prefix caching off (requests are unique).
- **Scenario B — long-decode TPOT** (r2-frieren): TPOT scales directly with
  acceptance rate of speculative decoding for output-heavy workloads.
  Hypothesis: vLLM + aggressive ngram speculative decoding (5-7 spec tokens) +
  FP8 KV cache + FlashInfer attention.
- **Scenario C — concurrent throughput** (r2-tanjiro): Throughput is bounded by
  effective batch size and KV memory.
  Hypothesis: vLLM + FP8 quantization + FP8 KV cache + large `--max-num-seqs`
  (256+) + FlashInfer attention + tuned chunked-prefill size.

Each student produces a reusable launcher recipe under
`senpai/launchers/<scenario>/<slug>/start_server.sh` plus a full-eval
`metrics.json` and W&B run logged under group `ib-20260522-r2-round1`.

## GPU coordination

3 logical students share 1 GPU. Heavy benchmark runs must be staggered.
Each student is told to coordinate by checking pod GPU state before launching a
benchmark, prep their launcher and any quick checks first, then take the GPU
slot when free.

## Potential next research directions

- Engine A/B: SGLang vs vLLM under the same scenario constraints once one
  vLLM-only winner is locked.
- Speculative decoding sweep on scenario D: a TPOT-leaning version of D may
  benefit from spec decoding similarly to B.
- KV cache dtype ablation: pure auto vs fp8 vs fp8_e4m3 on each scenario.
- Attention backend ablation: FLASH_ATTN vs FLASHINFER vs TRITON_ATTN at the
  best per-scenario operating point.
- Chunked-prefill size sweep specifically for scenario A (8192-token inputs).
- Quality-gate-safe quantization (FP8 weights vs FP8 kv only) — confirm MMLU-Pro
  pass ratio remains >= 0.95 of baseline for any FP8 winner.
- Mature-winner cross-scenario confirmation (A-D geomean) once any scenario
  beats its reference number.
