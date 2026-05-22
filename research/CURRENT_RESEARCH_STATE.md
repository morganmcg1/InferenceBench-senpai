# SENPAI Research State

- **Timestamp:** 2026-05-22 (last updated 22:57 UTC)
- **Latest direction from human research team:** No active human directives.
  This is the initial wave of the ib-20260522-r4 research program.
- **Advisor branch:** `ib-20260522-r4-advisor` (PRs target this, students
  branch from this, winners merge back to this).
- **W&B project:** `wandb-applied-ai-team/inferencebench-senpai`.
- **Hard constraints:** 1 H100 80GB in pod (shared by 3 students),
  Mistral-7B-Instruct-v0.3 base model, MMLU-Pro τ=0.95 quality gate,
  OpenAI-compatible foreground launcher contract, ~2 hour research window.

## Current research focus

The first wave of three near-orthogonal launcher hypotheses, one per scenario,
chosen to cover the high-headroom space and confirm the engine-tuning surface
quickly:

- **Scenario B (output-heavy, baseline to beat: 15.23x SMAC3):** PR #14,
  r4-frieren — vLLM with n-gram speculative decoding (`num_speculative_tokens=5,
  prompt_lookup_max=4`) + FP8 KV cache + FLASH_ATTN. Highest expected value
  this round; concurrency 1 with 8192-token decoding makes speculative
  decoding nearly all-upside on LongBench-v2 prompts.
- **Scenario A (input-heavy, baseline to beat: 4.37x SMAC3):** PR #15,
  r4-fern — vLLM with FLASHINFER attention backend + FP8 weights + FP8 KV
  cache, chunked prefill disabled (concurrency 1 gets no overlap benefit), CUDA
  graphs on.
- **Scenario C (high-load, baseline to beat: 51.12x SGLang default):** PR #16,
  r4-tanjiro — **PIVOTED 22:57 UTC** from SGLang to vLLM throughput-aggressive
  (max-num-seqs=256, max-num-batched-tokens=16384, FP8 KV cache, block-size=32,
  chunked-prefill on, prefix caching on). Pivot reason: pod ships vLLM 0.11.0
  but not sglang; installing sglang would clobber flash-attn-4 / flashinfer /
  transformers / openai and risk corrupting the other two students' in-flight
  vLLM sessions. New target: beat 48.69x vLLM-default ref; reach for 51.12x
  SGLang-default reference.

## GPU sequencing plan (1 GPU, 3 students) — REVISED 22:57 UTC

Serial slots, coordinated by `SLOT-FREE` comments on each PR. Original plan
had r4-frieren in slot 1 but they stalled 21+ min with 0% GPU and 0 commits,
so the queue was reshuffled:

1. **r4-fern (Scenario A, PR #15)** — slot 1, promoted, told to go now.
2. **r4-frieren (Scenario B, PR #14)** — slot 2, demoted; must post status
   before they get the slot back. If they cannot launch by the time slot 2
   opens, PR will be closed and reassigned.
3. **r4-tanjiro (Scenario C, PR #16)** — slot 3 unchanged; doing prep with the
   pivoted vLLM throughput-aggressive launcher.

## Infrastructure notes discovered 22:50 UTC

- Pod has vLLM 0.11.0; **does not have sglang or TGI** installed and we cannot
  install sglang without disturbing in-flight vLLM sessions. All round-1 and
  near-term hypotheses must be **vLLM-only** unless a clean isolated venv path
  is opened up.
- No precomputed `pytorch_baseline_metrics.json` on the pod for any scenario.
  Students should log raw metrics to W&B and compare against the public
  reference snapshot (PyTorch=1.00x, vLLM-default per scenario) in their
  terminal `SENPAI-RESULT` comments. `--baseline-primary 1.0` is the
  workaround for `log_metrics_to_wandb.py`.
- Scenario task workspaces on this pod do **not** include `test_server.sh`.
  Students should background-launch `start_server.sh` directly and call
  `python evaluate.py --server-url ...`. This preserves the contract as long
  as the launcher itself is foreground (`exec ...`) and self-contained.

Each student does quick eval first, caps at 3 quick-eval arms within their
launcher family, only runs full eval if quick eval clearly clears their
scenario baseline, and posts a terminal `SENPAI-RESULT` either way.

## Potential next research directions (round 2)

Held in reserve from the round-1 researcher pass and from re-reading the
reference table. Order by expected value, conditioned on what round 1 returns:

- **Scenario D (5.69x SMAC3 baseline):** vLLM chunked-prefill interleaving at
  concurrency 4 to mitigate head-of-line blocking on a balanced workload.
- **Scenario D (alt engine):** TGI with FP8 + CUDA graphs (`--cuda-graphs
  1,2,4,8,16,32`); orthogonal engine; SMAC3 reference used vLLM/SGLang only.
- **Scenario B follow-up if ngram works:** push `num_speculative_tokens` up
  to 7 and/or pair with FP8 weights for additional decode acceleration.
- **Scenario A follow-up if FLASHINFER fails:** try TRITON_ATTN with FP8 KV
  cache; or try vLLM `--block-size 32` (larger blocks help long single-stream
  prefill memory layout).
- **Scenario C follow-up if SGLang aggressive wins:** test `chunked_prefill_size=4096`
  vs 8192; SGLang `--enable-torch-compile` (if startup time stays under budget).
- **AWQ/GPTQ quantization probe:** known to outperform FP8 on Mistral-7B for
  TTFT on H100 in some setups (Marlin kernel). Single-scenario probe on A.
- **CUDA graph capture range tuning:** vLLM `--cuda-graph-sizes` for B at
  concurrency 1 — fewer captured shapes = less startup time, no quality cost.
- **Engine alternative if vLLM/SGLang both plateau:** TensorRT-LLM if the
  container has it (verify before assigning).

## Decision rules for round 2

- Merge every round-1 winner sequentially, update `BASELINE.md`.
- For PRs that came close but did not beat baseline, send back with one
  specific variation rather than closing.
- For PRs that crashed or failed quality, close and reassign with a different
  engine for that scenario.
