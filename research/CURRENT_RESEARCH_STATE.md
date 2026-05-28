# SENPAI Research State — `ib-20260528-scen-d-r1`

- **Updated:** 2026-05-28 17:02 UTC (start gate 16:36:57 UTC, budget end ~18:36 UTC, ~94 min left)
- **Launch budget:** ~2 hours, single Scenario D.
- **Most recent human directive:** none in this launch (no open team issues).
- **GPU topology:** 1 RTX PRO 6000 Blackwell, shared by 2 logical students (scen-d-frieren, scen-d-fern). Coordinate via `senpai/gpu_slot.py`.

## Current focus

Scenario D (General workload, 4096in / 2048out, 96 requests, burst concurrency 4) speedup over the PyTorch baseline. RTX PRO 6000 shakedown: results are calibration evidence, not leaderboard-comparable until repeated on H100.

The first round is **engine-diversified**: one vLLM arm and one SGLang arm. Goal is to put a calibrated launcher floor on the board fast, then iterate.

## Round 1 portfolio (in flight)

| PR | Student | Engine | Latest signal | Next step |
|---|---|---|---|---|
| #155 | scen-d-frieren | vLLM tuned | Arm 1 quick **1.17x**, Arm 2 quick **1.25x** at quick-mode c=1 (4 requests). TPOT-p50 already 1.50x faster than PyTorch at c=1 — wider-batch / chunked-prefill / prefix-cache levers are dormant at c=1 by construction. | Approved Arm 2 full eval (c=4, n=96, MMLU-Pro n=500). Expected ~25 min. Arm 3 contingency: n-gram speculative decoding (`num_speculative_tokens=5`, `prompt_lookup_min=2`, `prompt_lookup_max=3`) if Arm 2 full < 2.0x. |
| #157 | scen-d-fern | SGLang | No PR activity since assignment at 16:44 UTC. Pod log silent (student Claude session running). Likely mid SGLang venv install + boot. | Pinged for status checkpoint at 17:02 UTC. Latest reasonable full-eval start: 17:50 UTC. Fallback path: vLLM Arm 3 if SGLang install genuinely fails. |

Both students must use `gpu_slot.py run --wait --mode {quick,full}`. Quick caps 15 min, full caps 60 min, `--min-remaining-s 1500` required for full eval. Post a `SLOT-FREE` line on release.

### Quick-mode caveat (worth recording for future advisor loops)

The default quick eval runs at `request_limit=4, concurrency=1`. At c=1, the wider-batch + chunked-prefill + prefix-caching levers of a tuned vLLM/SGLang recipe are fundamentally dormant: there is only one request in flight, so the batch scheduler has nothing to do. Quick mode is fine for detecting "does the server boot and approximately how fast is single-request decode" but should not be used to reject a tuned recipe before full eval, because the levers being tested are concurrency-activated. Full eval (c=4, n=96) is the first measurement that actually exercises Scenario D's lever space.

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
