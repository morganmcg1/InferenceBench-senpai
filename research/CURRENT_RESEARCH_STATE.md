# SENPAI Research State — ib-20260524-leasefix-r3

- **Date:** 2026-05-24 23:05 UTC
- **Hardware:** 1 × NVIDIA RTX PRO 6000 Blackwell (shakedown — NOT H100-comparable)
- **Budget:** 2 hour SENPAI window (~50 min remaining)
- **Active students:** frieren, fern, tanjiro (3 students sharing 1 GPU pod)
- **Most recent human directive:** none in this launch

## Current research focus

Round 1 (in flight) is establishing a measured vLLM baseline per scenario on
RTX PRO 6000 by attacking the four scenarios with the levers most likely to
help on each workload shape. Round 2 (just started) is the high-throughput
scenario.

### Round 1 status (2026-05-24 23:05 UTC)

| Scenario | Student | PR | Hypothesis | Status |
|---|---|---|---|---|
| D | tanjiro | #77 | chunked-prefill + CUDA graphs + prefix cache, max_num_seqs=16, max_num_batched_tokens=8192 | **MERGED 1.305x** ✓ |
| A | frieren | #85 | chunked-prefill + max_num_batched_tokens=16384, max_num_seqs=8 | stuck — no W&B run, advisor nudged again at 23:05 |
| B | fern | #88 | n-gram speculative decoding (num_spec=5) + CUDA graphs, max_num_seqs=8 | background script queued behind tanjiro's lease, GPU now free, advisor nudged at 23:05 |

### Round 2 status (2026-05-24 23:05 UTC)

| Scenario | Student | PR | Hypothesis | Status |
|---|---|---|---|---|
| C | tanjiro | #91 | high-throughput vLLM: max_num_seqs=256, max_num_batched_tokens=16384, chunked-prefill + prefix cache + CUDA graphs + FLASH_ATTN | assigned, awaiting student start |

The Round 2 C assignment is sized to cover the largest absolute headroom
(H100 default vLLM hit 48.69x on C). Round 2 is single-scenario per student
because we are constrained on the 2-hour window; if frieren/fern's pending
arms complete cleanly we may have time for one more Round 2 assignment per
student before the window closes.

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

- **Scenario C breakdown after #91** — if #91's high-throughput launcher beats
  baseline, sweep `block_size` ∈ {16, 32}, `max_num_batched_tokens` ∈ {8192,
  16384, 32768}, and traffic-profile-specific scheduler policy.
- **FP8 quantization on Blackwell** — once a non-quantized winner exists per
  scenario, try `--quantization fp8` and `--kv-cache-dtype fp8`. Round 1 D
  produced a tanjiro/D-balanced-fp8 branch that did not yield a W&B run —
  worth retrying with the merged tanjiro launcher as the base.
- **SGLang alternative** — radix-tree prefix cache and longer-prompt scheduling
  may help scenarios A and C; budget one student for a head-to-head once vLLM
  baselines exist on all four scenarios.
- **Speculative decoding variants** — beyond n-gram, try EAGLE/medusa/draft
  model speculation for B if n-gram already gives a measurable TPOT win.
- **CUDA-graph capture batch sizes** — vLLM defaults to a fixed set; explicit
  `--cuda-graph-capture-sizes` lists pinned to {1,2,4,8} (from tanjiro's
  follow-up notes on #77) should tighten TPOT on B and D.
- **Cross-scenario confirmation** — when a launcher beats baseline on a single
  scenario, run A-D aggregate to populate the geomean.
- **Wrapper PGID bug** — tanjiro's PR #77 hit a tooling-only issue where the
  wrapper killed its own PGID, taking `gpu_slot.py run` with it; tracked as a
  separate tooling-only follow-up, not blocking any scenario research.

## Open questions and risks

- FlashInfer is disabled by `runtime_env.sh` on this hardware. Do not re-enable
  it in round 1 unless a launcher explicitly tests Blackwell FlashInfer.
- The advisor must respect the `gpu_slot.py` interface and tell students never
  to use broad `pkill` cleanup.
- All `SENPAI-RESULT` claims of `speedup_over_pytorch` must come from the
  PyTorch baselines under `/mnt/new-pvc/inferencebench-senpai/scoring-assets/
  rtxpro6000-seed248`, not the H100 README table.
