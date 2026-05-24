# SENPAI Research State — ib-20260524-leasefix-r3

- **Date:** 2026-05-24
- **Hardware:** 1 × NVIDIA RTX PRO 6000 Blackwell (shakedown — NOT H100-comparable)
- **Budget:** 2 hour SENPAI window
- **Active students:** frieren, fern, tanjiro (3 students sharing 1 GPU pod)
- **Most recent human directive:** none in this launch

## Current research focus

This is the first SENPAI round on this advisor branch. The opening focus is to
produce a measured vLLM baseline for every scenario by attacking the three
scenarios with the largest expected headroom over default vLLM on the H100
reference table (A: 1.25x, B: 2.25x, D: 1.96x), while preserving the quality
gate and benchmark integrity rules.

Scenario C already shows 48.69x for default vLLM on H100 (very throughput-bound
at high concurrency) so a single dataset configuration is likely close to the
GPU compute limit — we will explore C in round 2 once the steerable scenarios
have a measured starting point. Aggregate cross-scenario confirmation is
reserved for proven winners.

Round 1 hypothesis families (one per student, one scenario per PR):

1. **frieren — Scenario A:** chunked-prefill + low concurrency vLLM. Optimize
   for `1/ttft.p50` at concurrency=1, 8192-token prefills.
2. **fern — Scenario B:** n-gram speculative decoding + CUDA graphs vLLM.
   Optimize for `1/tpot.p50` at concurrency=1, 8192-token decode.
3. **tanjiro — Scenario D:** balanced vLLM (chunked-prefill + CUDA graphs +
   prefix caching) at concurrency=4. Optimize for the burst geomean.

## GPU coordination

3 logical students share 1 physical GPU. They must use `senpai/gpu_slot.py run
--wait` to serialize heavy server/evaluator workloads, with TTL ~25 minutes and
explicit owner/PR/scenario fields. Order in this round:

1. tanjiro (D) — first heavy slot; D speedup uses three of four scenario metric
   levers (TTFT/TPOT/req/s), so an early measurement informs the others.
2. frieren (A) — second heavy slot.
3. fern (B) — third heavy slot; speculative-decoding launchers need a known-good
   vLLM base first.

While the GPU is occupied, idle students prepare their launcher, run smoke
tests in dry-run mode, validate quality samples paths, and inspect `runtime_env.sh`
for the FlashInfer/MMLU defaults that apply on this hardware.

## Potential next research directions

- **Scenario C** (concurrent throughput) — once a baseline measurement exists
  on RTX PRO 6000, sweep `max_num_seqs` ∈ {64, 128, 256}, `max_num_batched_tokens`
  ∈ {4096, 8192, 16384}, and `enable_prefix_caching` true/false. C-poisson and
  C-constant traffic profiles also stress scheduler policy.
- **FP8 quantization on Blackwell** — once a non-quantized winner exists, try
  `--quantization fp8` and `--kv-cache-dtype fp8` for compute and memory wins,
  guarded by the MMLU-Pro quality gate. Confirm Blackwell support of fp8 paths
  before assigning.
- **SGLang alternative** — radix-tree prefix cache and longer-prompt scheduling
  may help scenarios A and C; budget one student for a head-to-head once vLLM
  baselines exist.
- **Speculative decoding variants** — beyond n-gram, try EAGLE/medusa/draft
  model speculation for B if n-gram already gives a measurable TPOT win.
- **CUDA-graph capture batch sizes** — vLLM defaults to a fixed set; explicit
  `--compile-config` or graph-size lists may improve TPOT on B and D.
- **Cross-scenario confirmation** — when a launcher beats baseline on a single
  scenario, run A-D aggregate to populate the geomean.

## Open questions and risks

- FlashInfer is disabled by `runtime_env.sh` on this hardware. Do not re-enable
  it in round 1 unless a launcher explicitly tests Blackwell FlashInfer.
- The advisor must respect the `gpu_slot.py` interface and tell students never
  to use broad `pkill` cleanup.
- All `SENPAI-RESULT` claims of `speedup_over_pytorch` must come from the
  PyTorch baselines under `/mnt/new-pvc/inferencebench-senpai/scoring-assets/
  rtxpro6000-seed248`, not the H100 README table.
