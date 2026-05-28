# SENPAI Research State — `ib-20260528-scen-c-r1`

- Date: 2026-05-28 (updated ~T+112, launch closing — final scoreboard preserved)
- Active research tag: `ib-20260528-scen-c-r1`
- Advisor branch: `ib-20260528-scen-c-r1`
- Scope: Scenario C only (high-load, geomean throughput across burst/poisson/constant).
- Students: scen-c-frieren, scen-c-fern (1 GPU shared across both via `senpai/gpu_slot.py`).
- Hardware: 1× RTX PRO 6000 (shakedown mode).

## Most recent human research direction

- Operator launch: paper-parity Scenario C trial, ~2h budget, two logical students on one benchmark
  GPU. Require terminal results to come from supervised relaunch + `validate_result.py`.

## Live baseline

**23.98x** speedup_over_pytorch (Scenario C geomean). **PR #165 merged**. vLLM 0.11.0 + n-gram speculative decoding (num_speculative_tokens=5, prompt_lookup_min=3, prompt_lookup_max=5) on top of PR #161 Arm A config (max_num_seqs=384, max_num_batched_tokens=16384, gpu_mem_util=0.92, chunked_prefill ON). W&B: mj8f07f0. Quality ratio 1.000 (n=500, exact match). Self-contained launcher.

Lineage:
- Rank 1: PR #165 vLLM+ngram 23.98x (current)
- Rank 2: PR #163 SGLang 22.18x
- Rank 3: PR #161 vLLM Arm A 20.84x

## Research findings worth carrying forward

- **vLLM 0.11.0 V1: `--no-enable-chunked-prefill` is a no-op** (engine forces chunked_prefill=True in arg_utils.py:1548 for non-pooling models). Future scenarios A/B/D should not test this flag.
- **n-gram speculative decoding wins on long-decode regimes** with structured/instruction-tuned models — Scenario C's output_len=1024 + ignore_eos=true is the right environment. Quality preserved exactly (ratio 1.000).
- **SGLang is competitive with vLLM on RTX PRO 6000** when configured with triton attention + pytorch sampler. The FlashInfer/SM120 incompatibility doesn't prevent strong baselines.
- **SGLang launcher self-containment (PR #174 merged)**: PR #163's launcher previously fell back to non-winning defaults when env vars were unset; PR #174 hardcoded the PR #163 Arm A winning config (mem=0.88, max_running=256, fcfs, chunked_prefill=4096) as the unset-fallback so the launcher matches the PR #165 vLLM launcher's self-contained pattern. Env-var overrides preserved.
- **Operational hazard: stale SGLang servers bind port 8000 after GPU release** (surfaced by PR #175 / frieren). SGLang's `python -m sglang.launch_server` can leave a defunct scheduler subprocess and the parent process bound to port 8000 even after the GPU memory lease has been released. The next launcher's readiness probe (curl on port 8000) will falsely succeed against the orphan server, producing meaningless metrics. **Mitigations for future launches**: (1) add `fuser -k 8000/tcp || true` or `pkill -f "sglang.launch_server" || true` at the top of every `start_server.sh`; (2) make `gpu_slot.py` lease release explicitly kill child processes; or (3) cycle the port number per PR. Frieren's manual diagnosis under time pressure caught this; the next launch should not rely on manual catches.

## Current state

- ~T+112 of 120 min launch budget. ~8 min remaining; launch is in final scorekeeping.
- Final scoreboard:
  - PR #161 merged (rank 3, 20.84x — vLLM Arm A)
  - PR #163 merged (rank 2, 22.18x — SGLang Arm A)
  - PR #165 merged (**rank 1, 23.98x — FINAL WINNER**: vLLM 0.11.0 + n-gram speculative decoding k=5)
  - PR #167 closed (fern's SGLang concurrency push stalled — no commits before time ran out)
  - PR #174 merged (SGLang launcher recipe-preservation cleanup — no metric change)
  - PR #175 closed (frieren's k=7 quick probe — exploratory: 4.24x quick vs 4.07x quick for k=5, +4% quick-mode bump; not promoted to full eval per scope)
- Both students idle in final ~8 min. No new assignments — launch in wrap-up.

## Active PRs

_None._ Launch is closed.

## Potential next research directions (for future launches)

- Higher num_speculative_tokens (7, 9) — diminishing returns expected.
- vLLM with `method=eagle` if a Mistral-7B EAGLE draft checkpoint becomes available.
- SGLang + speculative decoding (EAGLE/MEDUSA) to combine the two leads.
- FP8 weight quantization (BF16 KV preserved) — could free VRAM for higher concurrency.
- Try the SM120-compatible flashinfer build that recent SGLang patches reportedly support.

## Stop rules / hard limits

- Reserve last 10-15 minutes for review + `BASELINE.md` update.
- No full evaluation may start unless `gpu_slot.py --mode full --min-remaining-s 1800` passes.
- Quality gate failures are not winners.
- All terminal updates must come through `validate_result.py`.
- Do NOT score Scenarios A, B, D in this launch.
