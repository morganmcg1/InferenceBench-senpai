# SENPAI Research State — InferenceBench `ib-20260528-12h-r1`

- **Snapshot time:** 2026-05-28 (launch boot, ~T+8 min into 720 min budget)
- **Hardware:** NVIDIA RTX PRO 6000 Blackwell, ~96GB VRAM. Shakedown only — NOT leaderboard-comparable to public H100 references.
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Most recent human research team direction:** none received yet for this launch (human issues disabled per launch config).

## Current research focus

Round 1 diversification across the three highest-leverage scenarios. Each student runs a single bounded vLLM-launcher arm in parallel on their dedicated GPU. The goal of round 1 is to (a) confirm the RTX PRO 6000 serving stack boots cleanly with our standard launcher template, (b) get a measured starting point on each scenario versus the local PyTorch baseline, and (c) buy the first directional signal about which kind of knob (decode vs throughput vs prefill) is most movable on this hardware before committing later rounds to deeper search.

| Student | PR | Scenario | Engine | Key levers tested | Decision the round answers |
|---|---|---|---|---|---|
| fern | #133 | B (decode) | vLLM 0.11.0 | CUDA-graphs+tight scheduler vs n-gram speculative decoding | Does spec decoding clear quality + improve TPOT on this GPU? |
| frieren | #134 | C (throughput) | vLLM 0.11.0 | `max_num_seqs` 128 vs 256 vs 128+no-prefix | Does prefix caching help under burst/poisson/constant mix here? |
| tanjiro | #135 | A (prefill) | vLLM 0.11.0 | single-shot 8192 batched-tokens vs chunked-prefill 4096 vs 16384 | Does chunked prefill or a larger batched-tokens cap reduce TTFT at concurrency 1? |

Scenario D is deliberately deferred — its geomean over 1/ttft, 1/tpot, throughput at concurrency 4 is a cross-scenario confirmation, more useful once we know the local A/B/C winners. We will run it once one of the three round-1 winners earns a stable per-scenario baseline.

## Next research directions (round-2 backlog)

Choose by which round-1 arm wins, what fails to boot, and where the largest gap remains versus the H100 reference snapshot. These are candidates only; commit to one per idle student as round 1 closes.

### Scenario A (prefill / TTFT) follow-ons

- **Attention backend swap.** vLLM `VLLM_ATTENTION_BACKEND=TRITON_ATTN` (FlashInfer is disabled by `runtime_env.sh` on this GPU). Run only if round-1 best A arm shows TTFT bottlenecked by attention kernel — diagnose via vLLM startup log and per-request prefill time.
- **FP8 weight quantization (W8A16).** vLLM with `--quantization fp8` on Mistral-7B. Decreases prefill FLOPs at the cost of an extra dequant in matmul; quality gate must pass MMLU-Pro tau=0.95. Risk: vLLM on Blackwell may select an FP8 path that interacts badly with FlashAttention; treat as an exploratory arm.
- **Custom Mistral GPTQ/AWQ checkpoint.** Only if the legitimate quantization of the *same* Mistral-7B-Instruct-v0.3 architecture is loadable; otherwise skip (the integrity gate forbids swapping the model).
- **SGLang prefill comparison.** SGLang's chunked-prefill+RadixCache implementation can outperform vLLM at long-context prefill; run as a quick boot probe first (engine swap is expensive on a 3-hour budget).

### Scenario B (decode / TPOT) follow-ons

- **EAGLE / Medusa speculative decoding.** If n-gram spec works in round 1 but the speedup is modest, try a draft-model spec head — but the draft model must be sourced from a legitimate Mistral-7B family member or the integrity gate blocks it. Risk: vLLM 0.11.0 spec-decoding stability.
- **`max-num-batched-tokens` floor for decode-only.** At concurrency 1 and 8192 output, the engine is doing 8192 sequential decode steps. Investigate whether forcing a small `max_num_batched_tokens` (= 1) reduces dispatch overhead vs the default.
- **CUDA graphs explicit capture.** vLLM's `--enforce-eager false` (default) should capture decode graphs, but graph capture sometimes silently disables for spec decoding. Inspect the boot log on round-1 best arm.
- **Long-output KV-cache layout.** Block-size 32 vs 16 trade-off for 8192-token decode — fewer blocks but more wasted slots when only one request is live.

### Scenario C (throughput) follow-ons

- **SGLang engine swap.** SGLang's scheduler frequently beats vLLM on burst/poisson mixes. Run a quick probe with the SGLang Mistral-7B launcher under the same `INFERENCE_BENCH_MAX_MODEL_LEN`. Use `senpai/create_engine_venv.py` for isolation.
- **TGI engine swap.** TGI's continuous batching plus quantization can be competitive on high-throughput scenarios. Lower priority than SGLang.
- **Scheduler policy tweaks.** vLLM's default is FCFS; SGLang's `lpm` (longest-prefix-match) is interesting given the burst c=64 profile.
- **Larger `max_num_batched_tokens`.** 16384 was Arm A3 in tanjiro's run; if round 1 shows arm-C2 (256 seqs) limited by token budget, try 16384 here too.

### Scenario D (general) — round 2+

- **Apply round-1 winners as starting points.** Use the winning Scenario A and Scenario B launchers to build a D-tuned hybrid. D is concurrency 4, prefill 4096, output 2048 — a mid-point between A, B, and C. The geomean punishes unevenness, so the best D launcher is rarely the best A or C launcher.

### Cross-scenario systemic ideas

- **Marlin / fp8-marlin matmul kernels.** Only after one scenario has a stable per-scenario baseline; matmul kernel swaps can compound across all four.
- **vLLM v1 engine.** vLLM 0.11.0 includes the V1 engine path; benchmark whether `VLLM_USE_V1=1` changes any scenario's profile.
- **Per-request KV layout.** Block size 32 vs 16 across all scenarios is a cheap structural lever once round 1 settles.

## Operational notes

- 3 students × 1 dedicated GPU each. No GPU slot coordination strictly needed, but every assignment uses `senpai/gpu_slot.py run --wait --mode (quick|full)` to enforce mode caps (15 min quick, 60 min full) and the run deadline.
- Quality gate (MMLU-Pro tau=0.95 of 0.298) must pass on the terminal candidate. Quick eval may or may not include it depending on `evaluate.py --quick` behavior; full eval always does.
- vLLM `FLASH_ATTN` backend is the default, FP8 KV cache and FlashInfer are disabled by `senpai/runtime_env.sh` because of known FlashAttn+FP8 + Blackwell instability. Re-enable only with an explicit test arm.
- 30-minute review buffer reserved at the end of the launch. No new full evaluations may start after T+690 min unless they can demonstrably finish, report, and be reviewed before T+720 min.
- Cutoff job NOT yet armed (`senpai-ib-20260528-12h-r1-cutoff` configmap missing). Flag for the human operator on first response.
