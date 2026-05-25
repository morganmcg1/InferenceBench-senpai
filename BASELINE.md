# SENPAI Baseline — ib-20260525-three2-r1

## Setting
- **Hardware:** 1x NVIDIA RTX PRO 6000 Blackwell (~96GB VRAM, shakedown — not leaderboard-comparable to H100)
- **Model:** mistralai/Mistral-7B-Instruct-v0.3
- **Budget:** 2 hours total research program window
- **Scoring assets:** imported from `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`
- **Preflight:** passes for scenarios A/B/C/D (2026-05-25 19:48 UTC)
- **W&B project:** `wandb-applied-ai-team/inferencebench-senpai`

## PyTorch Baseline (raw objective per scenario, this hardware)
PyTorch baseline metrics are present in `src/eval/inference/baselines/speed/torch/inference_scenario_*/.../baseline_metrics.json`. The PyTorch baseline raw objective per scenario gets value 1.00x by definition.

## Current best per scenario (terminal full-eval valid)
| Scenario | Best speedup_over_pytorch | PR | W&B run | Notes |
|---|---:|---|---|---|
| A: Input-heavy (TTFT) | _none yet_ | — | — | — |
| B: Output-heavy (TPOT) | _none yet_ | — | — | — |
| C: High-load (req/s geomean) | **21.85x** | #110 | m6pt9mpu | vLLM 0.11.0 --max-num-seqs 256 --max-num-batched-tokens 8192 --enable-chunked-prefill; quality 0.31/0.298 ratio 1.04 ✅ |
| D: General (geomean) | _none yet_ | — | — | — |

## Reference snapshot (paper, H100, NOT this hardware)
From `program.md` — public reference, do not rank as our baseline. Used only as ceiling guidance.
- A best 4.37x | B best 15.23x | C best 46.70x | D best 5.69x (SMAC3 2h vLLM)
- vLLM default (no agent): A 1.25x | B 2.25x | C 48.69x | D 1.96x
- SGLang default (no agent): A 1.22x | B 1.77x | C 51.12x | D 2.14x

## Scenario C detail — PR #110 (merged 2026-05-25 20:36 UTC)
- **Launcher:** `senpai/launchers/C/vllm-tuned-256seq-8192batch/start_server.sh`
- **Engine:** vLLM 0.11.0
- **Key flags:** `--max-num-seqs 256 --max-num-batched-tokens 8192 --enable-chunked-prefill --no-enable-prefix-caching --gpu-memory-utilization 0.92`
- **MAX_MODEL_LEN:** 32768 (set by runtime_env.sh)
- **Per-profile req/s:** burst 2.68, poisson 1.96, constant 1.21 (geomean 1.85 req/s)
- **PyTorch baseline geomean req/s:** 0.0847 req/s → speedup 21.85x
- **Quality:** MMLU-Pro 0.310 / 0.298 / ratio 1.04 ✅ | 768/768 success
- **VRAM peak:** 87 GiB / 96 GiB
- **W&B:** `wandb-applied-ai-team/inferencebench-senpai` run `m6pt9mpu`
- **Note:** SGLang blocked on SM120 wheels (RTX PRO 6000 Blackwell); vLLM fallback only.

## Update history
- 2026-05-25 19:48 UTC — initial scaffold created; no terminal winners yet.
- 2026-05-25 20:36 UTC — Scenario C baseline set at 21.85x from PR #110 (tanjiro); first terminal winner.
