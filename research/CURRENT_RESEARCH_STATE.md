# SENPAI Research State — `ib-20260528-scen-d-r1`

- **Updated:** 2026-05-28 16:44 UTC
- **Launch budget:** ~2 hours, single Scenario D.
- **Most recent human directive:** none in this launch (no open team issues).
- **GPU topology:** 1 RTX PRO 6000 Blackwell, shared by 2 logical students (scen-d-frieren, scen-d-fern). Coordinate via `senpai/gpu_slot.py`.

## Current focus

Scenario D (General workload, 4096in / 2048out, 96 requests, burst concurrency 4) speedup over the PyTorch baseline. RTX PRO 6000 shakedown: results are calibration evidence, not leaderboard-comparable until repeated on H100.

The first round is **engine-diversified**: one vLLM arm and one SGLang arm. Goal is to put a calibrated launcher floor on the board fast, then iterate.

## Round 1 portfolio (in flight)

| PR | Student | Engine | Lever family | Decision tree gate | Notes |
|---|---|---|---|---|---|
| #155 | scen-d-frieren | vLLM | chunked prefill + prefix cache + wider batch (max-num-seqs 32, max-num-batched-tokens 8192), FLASH_ATTN (no FlashInfer), CUDA graphs ON, gpu-mem 0.92 | ≥3x quick → full; 1.5–3x → Arm 2 wider batch; <1.5x → stop after Arm 2 | RTX PRO 6000 floor for tuned vLLM. |
| #157 | scen-d-fern | SGLang (per-PR venv) | LPM scheduler + chunked-prefill 4096 + mem-fraction 0.85 + triton attention | ≥3x quick → full; 1.5–3x → quick FCFS vs LPM A/B; install fails → vLLM Arm 3 fallback | Engine diversity. RadixAttention prefix cache vs vLLM's. |

Both students must use `gpu_slot.py run --wait --mode {quick,full}`. Quick caps 15 min, full caps 60 min, `--min-remaining-s 1500` required for full eval. Post a `SLOT-FREE` line on release.

## Next likely directions (round 2 candidates)

Pick the surviving best engine from round 1 and broaden. Hold each as one bounded research arm if a student becomes idle within the launch window:

- **Round-2A — speculative decoding (n-gram or EAGLE-3) on the winning engine.** Big lever on general 4096in/2048out workloads. Likely worth a single full eval if quality holds.
- **Round-2B — KV-cache dtype FP8 / kv-cache-dtype auto comparison** on the winning engine. Only after round 1 confirms it boots cleanly here.
- **Round-2C — `enforce-eager` true vs CUDA-graphs ON A/B**. Quick screen to confirm CUDA graphs are net-positive on RTX PRO 6000 Blackwell at concurrency 4.
- **Round-2D — vLLM-vs-SGLang head-to-head** with both running the same prefix-cache + chunked-prefill recipe at the same memory fraction, to isolate engine effect from tuning effect.
- **Round-2E — TGI or TensorRT-LLM exploratory probe** only if the round-1 leader is well above 3x and we have wall time to spare; otherwise defer.

## Risks the advisor is tracking

- FlashInfer on RTX PRO 6000 Blackwell is known-flaky; `senpai/runtime_env.sh` disables it. Re-enabling it is out of scope unless we explicitly screen the path.
- FP8 KV cache may not behave well on this GPU class; only screen after the FP16/auto floor is established.
- Shared GPU + per-PR venv (SGLang) on the same Python image — fern must not `pip install --system` and must teardown her server cleanly so frieren's measurements stay valid.
- The 2-hour deadline is firm. Reserve the final 10–15 min for review, validate, and BASELINE update.
