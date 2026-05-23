# SENPAI Research State — InferenceBench ib-20260523-rerun-r3

- **Date/time:** 2026-05-23 10:15 UTC (advisor re-entry, boot + 22 min).
- **Most recent direction from human researcher team:** none — no open
  issues tagged `team` or `ib-20260523-rerun-r3-advisor` at boot.
- **Research tag:** `ib-20260523-rerun-r3`
- **Advisor branch:** `ib-20260523-rerun-r3-advisor`
- **Students:** r3-frieren, r3-fern, r3-tanjiro (1 GPU shared in pod)
- **Hardware:** NVIDIA RTX PRO 6000 Blackwell ~96 GB (shakedown — not yet
  leaderboard-comparable; H100 settings needed for paper claims)
- **Base model:** mistralai/Mistral-7B-Instruct-v0.3
- **Program budget:** 2 hours total

## Current research focus

Open the SENPAI shakedown for InferenceBench. There is no prior measured
SENPAI launcher on this advisor branch yet (BASELINE.md is bootstrapping).
The first goal of the run is to produce a first valid measured
`speedup_over_pytorch` per scenario, beat the vLLM default on each, and
calibrate which levers transfer cleanly from H100 references to the
Blackwell-class RTX PRO 6000.

Round 1 hypotheses (3 parallel PRs, sequential GPU slots in 1 pod):

| Slot | Student | Scenario | Hypothesis | PR |
|---|---|---|---|---|
| 1 (immediate) | r3-tanjiro | C high-load | vLLM FP8 weights + FP8 KV + FlashInfer + max-num-seqs 256, no prefix caching | #28 |
| 2 (after #28 SLOT-FREE) | r3-frieren | A input-heavy | vLLM FP16 weights + FP8 KV + FlashInfer + chunked prefill (16384 batched tokens) + prefix caching | #30 |
| 3 (after #30 SLOT-FREE) | r3-fern | B output-heavy | vLLM FP16 weights + FP8 KV + FlashInfer + n-gram speculative decoding (5 tokens) | #32 |

These three hypotheses span the three different metric directions of the
benchmark (throughput, prefill latency, decode latency) and target the three
most well-known orthogonal lever categories (quantization + scheduler
capacity, attention backend + chunked prefill, speculative decoding). Each
PR is a single-arm hypothesis with a documented fallback so the slot does not
idle on a flag-spelling mistake.

## Potential next research directions

When round 1 results come back, the strongest candidates for round 2 will
depend on which levers move the needle on Blackwell vs. the H100 references.
Likely follow-ups by scenario:

**Scenario A (long-context prefill / TTFT):**
- Sweep `max_num_batched_tokens` ∈ {8192, 12288, 16384, 24576} for the
  winning A launcher.
- Add `--num-scheduler-steps 4` or step-batching if vLLM exposes it.
- Try SGLang RadixAttention prefix caching for Sc. A.
- Try TensorRT-LLM if compile time fits the budget.
- Block-size sweep (16 vs 32) — block 32 sometimes helps long prefill.

**Scenario B (long-decode / TPOT):**
- If n-gram speculative wins, sweep `num_speculative_tokens` ∈ {3, 5, 7, 9}.
- Try a draft model (e.g., a tiny 0.5B–1B helper) if memory allows;
  speculative acceptance rates rise sharply with a good draft model.
- Try EAGLE / Medusa drafting if vLLM build supports them.
- FP8 *weight* quantization on top of speculative — TPOT-bound workloads
  benefit from compressed weights.

**Scenario C (high-load throughput):**
- If FP8 weights fail quality, fall back to AWQ/GPTQ INT4 of Mistral-7B and
  see whether throughput rises while quality holds.
- Sweep `max-num-seqs` ∈ {128, 256, 384, 512} once we know KV memory fits.
- SGLang with continuous batching + radix-attention prefix caching across
  burst prompts.
- Increase `max_num_batched_tokens` to fully saturate prefill while
  scheduler is decoding.

**Scenario D (general balanced):**
- Confirmation run for a mature winner across A and B (geomean target).
- Test whether the same launcher recipe transfers across D's mid-length
  inputs (4096 in, 2048 out, concurrency 4).

**Cross-cutting:**
- CUDA-graphs sweep: ensure no `--enforce-eager` slips in; on Blackwell
  graphs sometimes need batch-size whitelist tuning.
- Engine bake-off: once vLLM has been pushed, try SGLang on the same
  recipes — sometimes a 10–20% delta appears.
- Speculative-decoding draft model bake-off if the n-gram experiment shows
  speculative engagement is the dominant TPOT lever.

## Risks tracked

- Preflight may fail (PyTorch baseline + request files not yet materialized
  on the pod). Each student instructed to materialize_requests if needed.
- FP8 weight quantization may fail MMLU-Pro 0.95 gate. Fallback documented
  in r3-tanjiro PR body.
- `--speculative-config` JSON form may not be accepted by the installed
  vLLM; r3-fern PR body documents the legacy-flag fallback.
- Single GPU shared across 3 students — slot ordering and SLOT-FREE
  signalling are explicit in every PR.

## Plateau status

Not applicable — this is round 1 of the launch. No measured baseline yet.

## Round 1 progress (10:15 UTC)

- All 3 student assignments picked up by the shared GPU pod between
  09:58:49 and 09:59:37. Each student switched onto its branch and the
  entrypoint launched a Claude iteration with the heartbeat prompt.
- Pod stdout has been silent since 09:59:37, which is the expected pattern
  when all three students are actively running their Claude Code iterations
  (per-iteration logs are written to `student_logs/iteration_*.log` inside
  the pod, not to stdout).
- No new commits or comments on PRs #28, #30, #32 yet.
- No W&B runs from the three students yet — none in groups
  `ib-r3-scC-fp8-highconc`, `ib-r3-scA-flashinfer-chunked`, or
  `ib-r3-scB-ngram-spec`.
- GPU utilization 0% as of the last visible heartbeat. Most likely
  explanations: r3-tanjiro (Slot 1) is still in preflight /
  `materialize_requests` / PyTorch baseline build, or vLLM is still
  warming up before quick eval.
- Entrypoint flagged PR #28 as `stale_wip` (no PR-side activity for ~14
  min). Not posting a check-in comment yet because that would add noise to
  what looks like normal early-setup latency. If after the next re-entry
  cycle there is still zero W&B activity AND no PR comments, leave a brief
  status-check comment on #28 asking r3-tanjiro for a heartbeat.
