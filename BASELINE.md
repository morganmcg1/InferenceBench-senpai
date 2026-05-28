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
| 1 | #165 | `senpai/launchers/C/vllm-frieren-arm2-ngram/start_server.sh` | vLLM 0.11.0 | **23.98x** | PASS (1.000) | mj8f07f0 | Arm A + n-gram speculative decoding (num_speculative_tokens=5, prompt_lookup_min=3, prompt_lookup_max=5) |
| 2 | #163 | `senpai/launchers/C/sglang-fern/start_server.sh` | SGLang 0.5.9 | 22.18x | PASS (1.054) | b4xhsfby | triton attn + pytorch sampler, mem_fraction_static=0.88, max_running_requests=256, fcfs, chunked_prefill_size=4096 |
| 3 | #161 | `senpai/launchers/C/vllm-frieren/start_server.sh` | vLLM | 20.84x | PASS (1.027) | btpqa2rl | max_num_seqs=384, max_num_batched_tokens=16384, gpu_mem_util=0.92, chunked_prefill ON, BF16 KV |

**Reproduce current best (PR #165, vLLM + n-gram speculation):**
The launcher is self-contained (no env-var dependency). Copy `senpai/launchers/C/vllm-frieren-arm2-ngram/start_server.sh` into a Scenario C task workspace, run `evaluate.py`. Key flags: `--max-num-seqs 384 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.92 --enable-chunked-prefill --no-enable-prefix-caching --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":5,"prompt_lookup_min":3}' --kv-cache-dtype auto`.

Notable: scenario C's `output_len=1024 ignore_eos=true` regime + Mistral-Instruct chat structure produces enough repeated token spans that n-gram lookup amortizes TPOT effectively. Quality is identical to torch baseline (ratio=1.000 at n=500) — speculation is exact, not approximate.

**Cross-engine observation**: vLLM's `--no-enable-chunked-prefill` CLI flag is a **no-op in vLLM 0.11.0 V1** — the engine unconditionally sets `enable_chunked_prefill=True` for non-pooling tasks in `arg_utils.py` line 1548. Quick-mode A-vs-D delta in PR #161 was therefore run-to-run noise, not a chunked-prefill effect. Documented for future Scenario A/B/D runs.

## Provisional / quick-only candidates (not yet terminal)

| PR | Engine | Quick speedup | Notes |
|---|---|---:|---|
| #165 Arm E quick | vLLM + n-gram spec, k=5 | 4.07x quick | Promoted to full eval — became current winner at 23.98x |
| #161 Arm D | vLLM (chunked-prefill OFF flag) | n/a | Flag is no-op in vLLM 0.11.0 V1 — discarded |

## Failed launches and dead ends

_None yet._

## Update history

- 2026-05-28 17:54: **PR #165 merged** — vLLM + n-gram speculative decoding wins with 23.98x (+8.1% over SGLang). Quality ratio 1.000 (n=500, exactly matches torch baseline). 768/768 succeeded. Launcher is self-contained — no env-var dependency. W&B: mj8f07f0.
- 2026-05-28 17:31: **PR #163 merged** — SGLang takes the lead with 22.18x (+6.4% over vLLM Arm A). Quality PASS (ratio 1.054, n=500, observed=0.314), 768/768 requests, validation_pass=true. W&B: b4xhsfby.
- 2026-05-28 17:19: **PR #161 merged** — first terminal Scenario C result. vLLM Arm A full eval: 20.84x speedup, quality PASS (ratio=1.027, n=500, MMLU-Pro observed=0.306), 768/768 requests, validation_pass=true. W&B: btpqa2rl.
- 2026-05-28: Created ledger at start of `ib-20260528-scen-c-r1`. Preflight passed against
  `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`.
