# SENPAI Research State — `ib-20260528-scen-d-r1`

- **Updated:** 2026-05-28 17:33 UTC (start gate 16:36:57 UTC, budget end ~18:36 UTC, ~63 min left; review at 18:20 UTC, latest full-eval start ~17:50 UTC)
- **MAJOR FINDING:** Scenario D speed eval runs at concurrency=1 because `src/eval/inference/runner.py:1159` defaults profile-level concurrency to 1 when scenario.json has only a `profile` dict (no `profiles` list). Scenario D's profile is `{"name":"burst","pattern":"burst"}` with no concurrency. The PyTorch baseline_metrics.json also only has the c=1 burst profile, so speedups vs baseline are apples-to-apples. Wider-batch / chunked-prefill / prefix-cache levers are dormant; per-token decode levers (speculative decoding, quantization, decoder kernels) dominate.
- **Launch budget:** ~2 hours, single Scenario D.
- **Most recent human directive:** none in this launch (no open team issues).
- **GPU topology:** 1 RTX PRO 6000 Blackwell, shared by 2 logical students (scen-d-frieren, scen-d-fern). Coordinate via `senpai/gpu_slot.py`.

## Current focus

Scenario D (General workload, 4096in / 2048out, 96 requests, burst concurrency 4) speedup over the PyTorch baseline. RTX PRO 6000 shakedown: results are calibration evidence, not leaderboard-comparable until repeated on H100.

The first round is **engine-diversified**: one vLLM arm and one SGLang arm. Goal is to put a calibrated launcher floor on the board fast, then iterate.

## Round 1 portfolio (in flight)

| PR | Student | Engine | Latest signal | Next step |
|---|---|---|---|---|
| #155 | scen-d-frieren | vLLM tuned | Arm 1 quick **1.17x**, Arm 2 quick **1.25x**. Claimed Arm 2 full eval at 16:58 UTC. No W&B full-mode run visible yet at 17:33 UTC. Expected terminal 17:30-17:50 UTC if eval is actually running (c=1 full ≈ 30-35 min). Pinged for heartbeat at 17:31. | Wait for terminal SENPAI-RESULT or heartbeat. If no signal by 17:45 UTC, declare stood-down and transfer slot to fern. |
| #157 | scen-d-fern | SGLang | Arm 1 LPM quick **1.168x** at c=1 (gen throughput 60.93 tok/s vs PyTorch 38.21 = +59% — best per-token decoder seen so far). Identified the c=1 harness behavior. Silent since 17:07. Pinged at 17:31 for spec-launcher prep status. | Should be prepping SGLang+NGRAM and vLLM-spec fallback launchers while frieren holds the slot. Ready to take the slot the moment frieren posts SLOT-FREE. |

Both students must use `gpu_slot.py run --wait --mode {quick,full}`. Quick caps 15 min, full caps 60 min, `--min-remaining-s 1500` required for full eval. Post a `SLOT-FREE` line on release.

### Concurrency-1 caveat (now load-bearing for the launch)

The scenario.json `concurrency: 4` field is NOT read by the speed evaluator. `runner.py:1159` falls back to `config.get('profile', ...)` and Scenario D's profile dict has no `concurrency`, so default=1. This means BOTH quick AND full eval run at c=1 for Scenario D. Wider-batch / chunked-prefill / prefix-cache levers are dormant at c=1 by construction. The c=1 PyTorch baseline (38.21 tok/s gen, 0.0382 req/s) is the real comparison.

This is not a bug we should fix — `scenario.json` is a protected benchmark file, and editing it would change the metric semantics. We just optimize for the actual c=1 scoring function.

### What actually moves the metric at c=1

1. **n-gram (lookup) speculative decoding** — biggest single lever at c=1, no draft model needed. Both vLLM (`--speculative-config '{"method":"ngram",...}'`) and SGLang (`--speculative-algorithm NGRAM ...`) support it.
2. **Decoder kernel choice** — SGLang's triton kernel already gives +59% gen throughput vs PyTorch baseline (best signal seen so far). vLLM's FLASH_ATTN gave +51%.
3. **CUDA graphs ON** — already enabled in both arms.
4. **FP8 quantization** — potential 2x decode at quality risk. Defer until budget allows.

## Next likely directions (round 2 candidates, budget-permitting)

Most of these will only be reachable if round-1 closes before t≈17:50 with a clearly winning engine and ≥30 min remaining. Otherwise, treat as deferred.

- **Round-2A — n-gram speculative decoding on the winning engine.** Already promised to frieren as an in-PR Arm 3 contingency if Arm 2 full < 2.0x.
- **Round-2B — vLLM-vs-SGLang head-to-head** with identical chunked-prefill + prefix-cache + memory-fraction settings, to isolate engine effect from tuning effect (only useful if both round-1 arms ship terminal results).
- **Round-2C — block-size 32 vs 16** on the winning vLLM recipe (smaller probe, may not be worth a slot lease).
- **Round-2D — KV-cache dtype FP8 on the winning engine.** Only if FP16 floor is well above 2.0x AND we know FP8 boots cleanly on RTX PRO 6000. Quality gate is the risk.
- **Round-2E — TGI or TensorRT-LLM exploratory probe** — deferred unless round 1 leader is ≥3x and ≥40 min remain.

## Risks the advisor is tracking

- FlashInfer on RTX PRO 6000 Blackwell is known-flaky; `senpai/runtime_env.sh` disables it. Re-enabling it is out of scope unless we explicitly screen the path.
- FP8 KV cache may not behave well on this GPU class; only screen after the FP16/auto floor is established.
- Shared GPU + per-PR venv (SGLang) on the same Python image — fern must not `pip install --system` and must teardown her server cleanly so frieren's measurements stay valid.
- The 2-hour deadline is firm. Reserve the final 10–15 min for review, validate, and BASELINE update.
