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

No SENPAI launcher has been measured yet on this branch / this hardware. Update
each row when the first terminal `SENPAI-RESULT` with full eval and clean
relaunch arrives.

| Scenario | Best launcher path | Engine | Primary metric (speedup over PyTorch) | Raw objective | Quality MMLU-Pro ratio | W&B run | PR |
|----------|--------------------|--------|--------------------------------------:|--------------:|-----------------------:|---------|----|
| A        | _pending_          | _-_    | _-_                                   | _-_           | _-_                    | _-_     | _-_|
| B        | _pending_          | _-_    | _-_                                   | _-_           | _-_                    | _-_     | _-_|
| C        | _pending_          | _-_    | _-_                                   | _-_           | _-_                    | _-_     | _-_|
| D        | _pending_          | _-_    | _-_                                   | _-_           | _-_                    | _-_     | _-_|

PyTorch baseline raw objectives come from the asset bundle imported via
`senpai/require_scoring_preflight.sh --import-dir
/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`.

## Update history

- 2026-05-24 — Advisor initialised the ledger for the
  `ib-20260524-leasefix-r1` run. Three idle students; no SENPAI measurements
  yet on this hardware/branch.
