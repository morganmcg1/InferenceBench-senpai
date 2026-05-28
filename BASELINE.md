# SENPAI Baseline Ledger — `ib-20260528-scen-c-r1`

Scope: **Scenario C only** (high-load, geomean request throughput across burst/poisson/constant profiles).

## Run setting

- Hardware: 1× NVIDIA RTX PRO 6000 Blackwell (~96GB VRAM) — shakedown mode, not leaderboard-comparable.
- Base model: `mistralai/Mistral-7B-Instruct-v0.3`.
- Time budget: ~2 hours from start gate (shared 1 GPU across 2 students).
- Primary metric: `scenario/C/speedup_over_pytorch` (geomean over burst/poisson/constant `request_throughput_req_per_s`).
- Quality gate: MMLU-Pro ratio ≥ 0.95 vs PyTorch baseline. Direction: speedup higher is better; quality is gate not metric.

## PyTorch baseline source (Scenario C)

- `src/eval/inference/baselines/speed/torch/inference_scenario_c_high_load/mistralai_Mistral-7B-Instruct-v0.3/baseline_metrics.json`
- Per-profile `request_throughput_req_per_s`: burst 0.0845, poisson 0.0847, constant 0.0849. Geomean ≈ 0.0848 req/s.
- MMLU-Pro torch baseline accuracy (seed=248, n=500): 0.298.

## Reference snapshot (paper, H100 80GB, 2h budget) — for orientation only

| Method | Sc. C req/s speedup |
|---|---:|
| SGLang default, no agent | 51.12x |
| vLLM default, no agent | 48.69x |
| SMAC3 search, 2h vLLM | 46.70x |
| TPE search, 2h vLLM | 43.46x |
| Random search, 2h vLLM | 41.81x |
| HF TGI default, no agent | 41.94x |
| Best agent, Claude Sonnet 4.6 | 33.93x |
| PyTorch baseline | 1.00x |

Reference points apply to H100 80GB at 2h, **not** RTX PRO 6000 shakedown. Treat as orientation, not target.

## Current best valid launcher (Scenario C)

| Rank | PR | Launcher | Engine | Geomean speedup | Quality (ratio) | W&B run | Notes |
|---:|---|---|---|---:|---|---|---|
| 1 | #161 | `senpai/launchers/C/vllm-frieren/start_server.sh` | vLLM | **20.84x** | PASS (1.027) | btpqa2rl | max_num_seqs=384, max_num_batched_tokens=16384, gpu_mem_util=0.92, chunked_prefill ON, BF16 KV |

**Reproduce:**
```bash
cd "$PROBLEM_DIR"
source senpai/runtime_env.sh
cp senpai/launchers/C/vllm-frieren/start_server.sh /tmp/workspace/start_server.sh
# Full eval: python evaluate.py --json-output-file metrics_full.json
```

Full vLLM command: `python3 -m vllm.entrypoints.openai.api_server --model $MODEL --max-model-len 32768 --gpu-memory-utilization 0.92 --max-num-seqs 384 --max-num-batched-tokens 16384 --kv-cache-dtype auto --enable-chunked-prefill --no-enable-prefix-caching`

## Provisional / quick-only candidates (not yet terminal)

| PR | Engine | Quick speedup | Notes |
|---|---|---:|---|
| #161 Arm D | vLLM (chunked-prefill OFF) | 3.80x quick | Burst TTFT collapsed in quick mode (+37% vs Arm A quick); student predicts this reverses in steady-state full eval — **needs full eval to confirm** |
| #163 Arm A0 | SGLang 0.5.9 defaults | 3.88x quick | Boot clean, quality passes; full eval pending |

## Failed launches and dead ends

_None yet._

## Update history

- 2026-05-28 17:19: **PR #161 merged** — first terminal Scenario C result. vLLM Arm A full eval: 20.84x speedup, quality PASS (ratio=1.027, n=500, MMLU-Pro observed=0.306), 768/768 requests, validation_pass=true. W&B: btpqa2rl.
- 2026-05-28: Created ledger at start of `ib-20260528-scen-c-r1`. Preflight passed against
  `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`.
