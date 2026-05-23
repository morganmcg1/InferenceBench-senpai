# Initial preflight state (r2-frieren PR #26, 2026-05-23)

Run from repo root:
```
source senpai/runtime_env.sh
python senpai/preflight.py --scenario all --expected-gpu "RTX PRO 6000" --require-wandb
```

Pass:
- hardware (RTX PRO 6000)
- wandb (api key + import)

Fail (expected — first run of this launch):
- speed_baseline_A / requests_A
- speed_baseline_B / requests_B
- speed_baseline_C / requests_C
- speed_baseline_D / requests_D
- quality_samples (mmlu_pro 248_500/samples.jsonl)
- quality_registry (torch quality baseline registry)

Warn:
- tokenizer_request_sampling: tokenizer not cached; set INFERENCE_BENCH_ALLOW_HF_DOWNLOAD=1 before materialize.

This file is r2-frieren's coordination note so other students see what was missing.
