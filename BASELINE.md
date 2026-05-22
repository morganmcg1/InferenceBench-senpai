# SENPAI InferenceBench Baseline (ib-20260522-r1)

Live advisor-owned baseline ledger. Update when a SENPAI PR with a clean terminal
result becomes the new best for its scenario.

## Run Setup

- Research tag: `ib-20260522-r1`
- Advisor branch: `ib-20260522-r1-advisor`
- Base model: `mistralai/Mistral-7B-Instruct-v0.3`
- **Actual pod hardware**: single **NVIDIA RTX PRO 6000 Blackwell (sm_120), ~97 GiB VRAM** shared by up to 3 logical students. (`program.md` reference snapshot assumes H100 80GB. The Blackwell pod runs the same eval contract but the absolute speedups in the snapshot are H100-calibrated; treat them as directional, not paper-grade, anchors on this pod.)
- Software: vLLM 0.21, PyTorch 2.11, CUDA 13.2 (vLLM wheel built against CUDA-12 ABI — needs `LD_LIBRARY_PATH` patch for the cu12 runtime). flashinfer JIT-compile fails on sm_120 in this pod (missing curand headers); use `VLLM_USE_FLASHINFER_SAMPLER=0` and pass `--attention-backend` as a CLI flag instead of `VLLM_ATTENTION_BACKEND` (the env var was dropped in vLLM 0.21).
- Baselines: pod did **not** ship with `pytorch_baseline_metrics.json` or the MMLU-Pro quality-baseline registry. r1-tanjiro (PR #4) is generating them with `precompute_quality_baseline.py --backend vllm` and a vLLM-default reference run.
- Time budget: 2h total wall-clock for the whole research program
- W&B: `wandb-applied-ai-team/inferencebench-senpai`

## Current Best (Live)

| Scenario | Primary metric | Current best speedup | PR | W&B run | Launcher path | Notes |
|---|---|---:|---|---|---|---|
| A: input-heavy | scenario/A/speedup_over_pytorch | — (no SENPAI run yet) | — | — | — | starting from `src/starting_points/vllm_running/start_server.sh` (vLLM default) |
| B: output-heavy | scenario/B/speedup_over_pytorch | — (no SENPAI run yet) | — | — | — | as above |
| C: high-load | scenario/C/speedup_over_pytorch | — (no SENPAI run yet) | — | — | — | as above |
| D: general | scenario/D/speedup_over_pytorch | — (no SENPAI run yet) | — | — | — | as above |

## Targets To Beat (Public Reference Snapshot, 2026-05-21)

These are public reference numbers from the InferenceBench README for Mistral-7B-Instruct-v0.3 on one H100 80GB with a 2h budget per run. They are stretch targets that SENPAI should aim to match or exceed; they are not the live baseline.

| Method | Aggregate | Sc. A TTFT | Sc. B TPOT | Sc. C req/s | Sc. D geomean |
|---|---:|---:|---:|---:|---:|
| SMAC3 search, 2h vLLM | 11.53x | 4.37x | 15.23x | 46.70x | 5.69x |
| TPE search, 2h vLLM | 11.25x | 4.48x | 14.76x | 43.46x | 5.58x |
| Random search, 2h vLLM | 10.20x | 4.21x | 11.34x | 41.81x | 5.42x |
| Best listed agent, Claude Sonnet 4.6 | 8.08x | 3.47x | 12.03x | 33.93x | 3.01x |
| vLLM default, no agent | 4.05x | 1.25x | 2.25x | 48.69x | 1.96x |
| SGLang default, no agent | 3.92x | 1.22x | 1.77x | 51.12x | 2.14x |
| HF TGI default, no agent | 3.30x | 1.14x | 1.37x | 41.94x | 1.80x |
| PyTorch baseline | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |

## Update History

- 2026-05-22 22:30 UTC — Initial advisor ledger created. No SENPAI candidate has yet been measured on this run; PyTorch=1.00x is the implicit floor and the vLLM default is the obvious quick reference.
- 2026-05-22 23:00 UTC — Hardware/software reality recorded. Pod is RTX PRO 6000 Blackwell sm_120, not H100. vLLM 0.21 quirks captured. Baseline-file blocker noted; r1-tanjiro generating quality + speed reference data via PR #4.
- 2026-05-22 23:35 UTC — Evaluator hotfix `69b0c3d` applied to advisor branch: `_count_chat_tokens` in `src/eval/inference/runner.py` was using `len(BatchEncoding)` which returned `2` on transformers ≥4.51, causing every LongBench-v2 sample to fail `min_input_tokens` filter and crash all four scenarios with "No LongBench-v2 samples can satisfy the input length range". 8-line patch reads `tokens.input_ids` when available. Reported independently by r1-frieren (PR #2) and r1-tanjiro (PR #4); co-credited to frieren. Both students should rebase onto advisor.
