# SENPAI Research State — `ib-20260528-scen-c-r1`

- Date: 2026-05-28
- Active research tag: `ib-20260528-scen-c-r1`
- Advisor branch: `ib-20260528-scen-c-r1`
- Scope: Scenario C only (high-load, geomean throughput across burst/poisson/constant).
- Students: scen-c-frieren, scen-c-fern (1 GPU shared across both via `senpai/gpu_slot.py`).
- Hardware: 1× RTX PRO 6000 (shakedown mode).

## Most recent human research direction

- Operator launch: paper-parity Scenario C trial, ~2h budget, two logical students on one benchmark
  GPU. Require terminal results to come from supervised relaunch + `validate_result.py`. No
  scoring or merging of A/B/D allowed.

## Current research focus

- Build the first Scenario C launcher portfolio across multiple engine families (vLLM, SGLang)
  before committing GPU budget to a single full evaluation.
- Establish a clean RTX PRO 6000 boot for at least one strong engine, then tune for high-concurrency
  throughput. The paper reference shows engine defaults are competitive on C (SGLang 51.12x,
  vLLM 48.69x) — beat the engine default, not just the PyTorch baseline.
- Validate that `gpu_slot.py` coordination keeps both students productive without overlapping
  GPU work.

## Initial portfolio

1. **scen-c-frieren — vLLM C throughput** (image-included engine, immediate GPU readiness).
   Tune `max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization`, BF16 KV cache, CUDA
   graphs on, FlashAttention backend. RTX PRO 6000 has ~96GB so push `gpu_memory_utilization`
   above stock 0.90.
2. **scen-c-fern — SGLang C throughput** (set up per-PR venv, then probe). Paper reference for
   SGLang default on C is highest of all engines (51.12x), so getting it to boot and pass
   quality is high-value.

## Potential next research directions (after first quick probes return)

- vLLM speculative decoding (n-gram or eagle) for high-concurrency decode-heavy workload.
- SGLang `--schedule-policy lpm` vs `fcfs` under burst concurrency.
- FP8 quantization on RTX PRO 6000 (the FP8 weight + BF16 KV combination, since vLLM
  FlashAttention rejects FP8 KV on RTX PRO 6000).
- Larger `--max-num-batched-tokens` (e.g. 32768) to better saturate burst concurrency.
- Engine candidate selection driven by measured quick result, not by personal preference.

## Stop rules / hard limits

- Reserve last 10-15 minutes for review + `BASELINE.md` update.
- No full evaluation may start unless `gpu_slot.py --mode full --min-remaining-s` succeeds.
- Quality gate failures are not winners.
- All terminal updates must come through `validate_result.py`.
