# InferenceBench SENPAI Live Baseline — ib-20260524-leasefix-r1

This is the advisor-owned live baseline ledger for the
`ib-20260524-leasefix-r1` research tag. Update when a terminal review-ready PR
beats the current entry for its scenario; old entries stay in the history
section.

## Run context

- **Research tag:** `ib-20260524-leasefix-r1`
- **Advisor branch:** `ib-20260524-leasefix-r1`
- **Target base branch (students fork from):** `ib-20260524-leasefix-r1`
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Hardware:** 1 × NVIDIA RTX PRO 6000 Blackwell (~96GB VRAM) — shakedown
  hardware. Not leaderboard-comparable until repeated on H100.
- **Per-run wall-clock budget:** 2 hours per scenario as documented in
  `program.md`; the advisor coordinates the shared GPU across students.
- **W&B project:** `wandb-applied-ai-team/inferencebench-senpai`
- **Pod scoping:** 3 students share 1 GPU. Heavy server/evaluator work must
  use `senpai/gpu_slot.py run --wait` to serialize.

## Reference snapshot (program.md, H100, public InferenceBench reference)

These are public H100 numbers from `program.md`; they bound the headroom for
the same scenarios on RTX PRO 6000 (which should run slower per arm but with
the same relative tuning shape):

| Method               | Sc. A TTFT | Sc. B TPOT | Sc. C req/s | Sc. D geomean |
|----------------------|-----------:|-----------:|------------:|--------------:|
| SMAC3, 2h vLLM       |       4.37 |      15.23 |       46.70 |          5.69 |
| TPE, 2h vLLM         |       4.48 |      14.76 |       43.46 |          5.58 |
| vLLM default         |       1.25 |       2.25 |       48.69 |          1.96 |

The "vLLM default" row is the contest baseline a student must clear before the
launcher is interesting.

## Current advisor-owned best launchers

| Scenario | Best launcher path | Engine | Primary metric (speedup over PyTorch) | Raw objective | Quality MMLU-Pro ratio | W&B run | PR |
|----------|--------------------|--------|--------------------------------------:|--------------:|-----------------------:|---------|----|
| A        | _pending_          | _-_    | _-_                                   | _-_           | _-_                    | _-_     | _-_|
| B        | `senpai/launchers/scenario_b/vllm-ngram5-fp8kv/start_server.sh` | vLLM 0.11 + n-gram-5 spec | **2.694x** | 1/tpot.p50 = 107.09 tok/s | 1.013 (0.302 obs / 0.298 base, n=500) | `rnc0c1by` + `jqilihl9` (relaunch) | #76 |
| C        | _pending_          | _-_    | _-_                                   | _-_           | _-_                    | _-_     | _-_|
| D        | _pending_          | _-_    | _-_                                   | _-_           | _-_                    | _-_     | _-_|

PyTorch baseline raw objectives come from the asset bundle imported via
`senpai/require_scoring_preflight.sh --import-dir
/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`.

## Update history

- 2026-05-24 23:24 UTC — **PR #76 (fern, Scenario B) merged.** First measured SENPAI baseline on this branch/hardware. vLLM 0.11 + n-gram-5 speculative decoding launcher delivers **2.694x speedup over PyTorch** on Sc B (full eval, 64 burst requests, 1K input / 8K output), passing MMLU-Pro quality gate at observed 0.302 (ratio 1.013, n=500). Beats the H100 vLLM-default reference (2.25x) by ~20% on RTX PRO 6000. Clean relaunch reproduced TPOT magnitude. Launcher uses `--speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'`, drops `--kv-cache-dtype fp8` (FA+FP8KV impossible on SM 12.0), uses `block_size=16`, `max_num_seqs=8`. W&B: `rnc0c1by` (full) + `jqilihl9` (clean-relaunch quick).
- 2026-05-24 — Advisor initialised the ledger for the
  `ib-20260524-leasefix-r1` run. Three idle students; no SENPAI measurements
  yet on this hardware/branch.
