# SENPAI BASELINE — ib-20260528-scen-a-r1

Live advisor-owned baseline ledger. Updated only from full clean-relaunch
results validated by `senpai/validate_result.py`.

## Run context

- **Research tag:** `ib-20260528-scen-a-r1`
- **Advisor branch:** `ib-20260528-scen-a-r1`
- **Hardware:** NVIDIA RTX PRO 6000 (~96 GB VRAM, shakedown — not H100 leaderboard-comparable)
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Scenario in scope:** A only (input-heavy, long-context prefill / TTFT)
- **Time budget:** 2h, gate opened `2026-05-28T16:36:57Z`, cutoff `2026-05-28T18:36:57Z`
- **Seed / scoring assets:** seed 248, imported from PVC
- **PyTorch speed baseline:** `src/eval/inference/baselines/speed/torch/inference_scenario_a_input_heavy/mistralai_Mistral-7B-Instruct-v0.3/baseline_metrics.json`
- **PyTorch quality baseline:** `src/eval/inference/baselines/quality/mistralai_Mistral-7B-Instruct-v0.3_torch.json` (MMLU-Pro accuracy ≈ 0.298, τ=0.95)

## PyTorch raw objective (for speedup conversion)

| Scenario | Profile | Raw metric | Value | Source |
|---|---|---|---:|---|
| A | burst | `ttft.p50` (s) | 0.4385 | baseline_metrics.json |
| A | burst | `1 / ttft.p50` (req/s, higher better) | 2.281 | derived |

## Current best valid launcher (full eval, validated)

| Scenario | Speedup over PyTorch | TTFT.p50 (s) | MMLU-Pro ratio | PR | W&B run | Launcher |
|---|---:|---:|---:|---|---|---|
| A | **1.8873x** | 0.2325 | 0.9597 (pass) | #160 | `bt2ckkzz` | `senpai/launchers/A/fern-fp8-weights/start_server.sh` |

## 2026-05-28 17:34 — PR #160: Scenario A vLLM FP8 weight quantization

- **Student:** scen-a-fern / branch `scen-a-fern/fern-fp8-weights`
- **Hypothesis:** On-the-fly vLLM FP8 weight quantization (`--quantization fp8`) halves
  GEMM bandwidth for the 8192-token prefill, yielding large TTFT.p50 reduction at
  burst concurrency 1 without changing the base model checkpoint.
- **Key metrics (full eval, 128 requests, seed 248, RTX PRO 6000):**

| Metric | Value |
|---|---:|
| `scenario/A/speedup_over_pytorch` | **1.8873x** |
| `scenario/A/inverse_ttft_p50` (raw objective) | 4.3038 req/s |
| PyTorch raw objective (`1/ttft.p50`) | 2.2804 req/s |
| TTFT.p50 (s) | 0.2325 |
| MMLU-Pro observed accuracy | 0.286 |
| MMLU-Pro baseline accuracy | 0.298 |
| MMLU-Pro ratio | 0.9597 (τ=0.95 → floor 0.283 → **pass**) |
| Speed: success / total | 128 / 128 |
| Speed: failure rate | 0.0 |
| VRAM peak (MB) | ~89,405 |

- **W&B run:** `bt2ckkzz` (group `fern-fp8-weights`, project `wandb-applied-ai-team/inferencebench-senpai`)
- **Validation:** `validation_pass=true`, `baseline_update_allowed=true` (via `senpai/validate_result.py`)
- **Launcher:** `senpai/launchers/A/fern-fp8-weights/start_server.sh`
- **Reproduce (from task workspace):**
  ```bash
  source "$PROBLEM_DIR/senpai/runtime_env.sh"
  WORKDIR=/tmp/inferencebench-scen-a-repro
  python "$PROBLEM_DIR/senpai/create_task_workspace.py" --scenario A \
    --output "$WORKDIR" --starting-point vllm_running --replace
  cp "$PROBLEM_DIR/senpai/launchers/A/fern-fp8-weights/start_server.sh" "$WORKDIR/start_server.sh"
  chmod +x "$WORKDIR/start_server.sh"
  cd "$WORKDIR"
  source ./eval_env.sh && ./clean_eval_artifacts.sh
  ./test_server.sh > agent/server.log 2>&1 &
  python evaluate.py --json-output-file metrics_full.json
  ```
- **Notes:**
  - `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_DISABLE_FLASHINFER_PREFILL=1`,
    `VLLM_DISABLE_FLASHINFER_SAMPLING=1` must be set explicitly inside the launcher
    (the runtime_env.sh defaults do not persist through the gpu_slot shell on
    this packed-pod image). These exports are embedded in the winning launcher.
  - Quality is close to the τ=0.95 floor (ratio 0.9597); further FP8 variants
    that reduce model fidelity may push quality below the gate.
  - BF16 vLLM default ceiling confirmed at ~1.276x (PR #158 A0/A1 neutral).
    FP8 weights are the dominant Scenario A lever on RTX PRO 6000.

## Reference snapshot (H100, paper-public)

| Method | Sc. A TTFT speedup |
|---|---:|
| SMAC3 search, 2h vLLM | 4.37x |
| TPE search, 2h vLLM | 4.48x |
| Best listed agent (Claude Sonnet 4.6) | 3.47x |
| vLLM default, no agent | 1.25x |
| PyTorch baseline | 1.00x |

These are paper-public H100 numbers and are not directly comparable to current
RTX PRO 6000 shakedown results. They indicate where the meaningful headroom is.

## Update history

- 2026-05-28T16:40Z — initialized BASELINE.md from preflight import (no current best yet).
- 2026-05-28T17:34Z — updated current best to PR #160 (scen-a-fern): 1.8873x speedup via FP8 weight quantization. Full eval 128/128 success, MMLU-Pro ratio 0.9597 (pass), W&B `bt2ckkzz`.
