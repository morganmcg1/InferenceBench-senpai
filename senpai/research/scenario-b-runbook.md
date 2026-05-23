# Scenario B candidate runbook — what to do when GPU is free

Operational checklist for executing the n-gram speculative decoding + FP8 KV
candidate on PR #22 once the upstream baselines are written. Pairs with the
deeper analysis in `scenario-b-ngram-spec-notes.md`.

## Current launch state (rolling)

- Slot 1 cut on PR #20; advisor transferred baseline build to r1-tanjiro on
  PR #23 at 10:49 UTC.
- r1-tanjiro is running `precompute_all_baselines.py` for scenarios B then A,
  with internal request sampling (bypasses tokenizer-roundtrip bug).
- I (r1-fern) am slot 2, holding for the SLOT-FREE signal that announces
  Scenario B speed baseline + MMLU-Pro quality registry on disk.

## Prep that is already done

- `senpai/launchers/B/ngram-spec-fp8-kv/start_server.sh` committed; flags
  parse cleanly under vLLM 0.21.0 (verified via argparse dry-run).
- Task workspace staged at `/tmp/ib-B/` with `start_server.sh` copied to
  `/tmp/ib-B/task/`.
- `/tmp/ib-B/run_eval.sh` wrapper written. Accepts
  `BASELINE_REPO=...` env var; routes baseline paths through env overrides
  (`INFERENCE_BENCH_REQUESTS_FILE`,
  `INFERENCE_BENCH_QUALITY_BASELINE_REGISTRY`,
  `INFERENCE_BENCH_QUALITY_BASELINE_BACKEND=torch`) and passes
  `--baseline-primary` to summarize/log scripts (works around the
  `senpai/summarize_metrics.py::_baseline_primary_value` wrapping bug —
  baseline JSON nests metrics under `"baseline"` but the helper expects them
  at top level).
- Quality samples cached at
  `src/eval/inference/baselines/samples/mmlu_pro/248_500/samples.jsonl`
  inside my repo (preflight PASS).

## Execution sequence the moment SLOT-FREE arrives

1. Read r1-tanjiro's SLOT-FREE comment on PR #23 to confirm the baseline
   path. Expected baseline repo root: `/workspace/senpai-r1-tanjiro/target`.
2. Verify the assets are on disk:
   ```bash
   ls /workspace/senpai-r1-tanjiro/target/src/eval/inference/baselines/speed/torch/inference_scenario_b_output_heavy/mistralai_Mistral-7B-Instruct-v0.3/baseline_metrics.json
   ls /workspace/senpai-r1-tanjiro/target/src/eval/inference/baselines/quality/mistralai_Mistral-7B-Instruct-v0.3_torch.json
   ```
3. Post `GPU-CLAIM: r1-fern scenario B quick eval` on PR #22.
4. Quick eval:
   ```bash
   BASELINE_REPO=/workspace/senpai-r1-tanjiro/target /tmp/ib-B/run_eval.sh quick
   ```
   Watch `agent/server.log` for vLLM startup errors. If quality gate fails
   on quick (MMLU-Pro ratio < 0.95), the spec-decode + FP8 stack is
   regressing accuracy; abort and report negative result.
5. Full eval (relaunch from fresh shell so we exercise the launcher
   contract once more):
   ```bash
   BASELINE_REPO=/workspace/senpai-r1-tanjiro/target /tmp/ib-B/run_eval.sh full
   ```
6. Capture VRAM peak, W&B run ID, the `SENPAI-RESULT` marker line printed
   by `summarize_metrics.py`, and the speculative acceptance rate from the
   server log (vLLM prints `spec_decode/<...>` if enabled).
7. Post the terminal `SENPAI-RESULT` comment on PR #22 using the template
   from `target/program.md`. Include:
   - launcher file contents + path
   - exact quick + full eval commands
   - metrics_full.json path under `/tmp/ib-B/task/`
   - MMLU-Pro observed/baseline/ratio
   - TTFT, TPOT, ITL, request/generation throughput, success counts
   - Peak VRAM (read via `nvidia-smi --query-gpu=memory.used --format=csv`
     while the server is hot)
   - Raw `1/tpot.p50` candidate vs. the torch baseline value, plus
     `scenario/B/speedup_over_pytorch`
   - Whether the candidate survived a clean relaunch
   - Honest analysis: did n-gram speculation help on this workload?
8. Tear down (`pkill` only your own server PGID; never broad). Post
   `SLOT-FREE: r1-fern done with GPU` on PR #22.

## Stop conditions / negative-result handling

- **Quality gate fails on quick eval:** report immediately as negative;
  do not waste the full-eval budget. Add suggested follow-up: drop
  `--speculative-config`, keep FP8 KV + CUDA graphs only.
- **OOM at startup:** Mistral-7B is ~14 GB fp16 + ~7 GB FP8 KV at 131 K
  context, plus n-gram draft buffer. On a 96 GB RTX PRO 6000 there is plenty
  of headroom, so an OOM here suggests an unexpected memory blow-up — lower
  `--gpu-memory-utilization` to 0.85 and rerun.
- **vLLM rejects `--speculative-config` JSON:** fall back once to the legacy
  `--speculative_model [ngram] --num-speculative-tokens 5
  --ngram-prompt-lookup-max 4 --ngram-prompt-lookup-min 2` form per PR body.
- **Time budget runs out before full eval:** submit the quick-eval result as
  partial (`terminal=false`, `pending_arms=true`) so the round's leaderboard
  at least sees something. The advisor's primary-metric contract still
  requires `--baseline-primary` on the partial.
