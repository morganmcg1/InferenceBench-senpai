# SENPAI Research State — ib-20260527-latest3-r1

- **As of:** 2026-05-27, run start
- **Last human directive:** none (no open GitHub Issues from research team)

## Research focus

InferenceBench LLM inference optimization. Goal: maximize per-scenario speedup
over PyTorch baseline on Mistral-7B-Instruct-v0.3 (RTX PRO 6000 shakedown, single
GPU, 2 hour budget) while passing the MMLU-Pro quality gate.

Reference table shows where the headroom lives:
- **A (input-heavy):** PyTorch 1.0x → vLLM default 1.25x → SMAC3 best 4.37x. Huge gap. Levers: chunked prefill, max-num-batched-tokens, prefix caching.
- **B (output-heavy):** PyTorch 1.0x → vLLM default 2.25x → SMAC3 best 15.23x. Biggest relative gap. Levers: CUDA graphs (no enforce-eager), prefix caching, small max-num-seqs for c=1 decode focus, possibly FP8 model quant.
- **C (high-load):** PyTorch 1.0x → vLLM default 48.69x → SMAC3 best 46.70x. Defaults already saturate; small wins via batched-tokens and mem fraction. SGLang default 51.12x is slightly ahead — alternative engine worth testing.
- **D (general):** PyTorch 1.0x → vLLM default 1.96x → SMAC3 best 5.69x. Balanced workload. Levers similar to A+B blend.

## Current portfolio (round 1)

Three idle students, one shared GPU. First assignments diversify across scenarios
and engines:

1. **frieren → Scenario B (vLLM, decode-heavy)** — CUDA graphs ON, prefix caching, low max-num-seqs. Highest absolute headroom.
2. **fern → Scenario A (vLLM, prefill-heavy)** — chunked prefill ON with large batched-tokens window.
3. **tanjiro → Scenario C (SGLang, alt-engine)** — SGLang default already beats vLLM default on C; quick tuning likely wins.

## Coordination

- Single shared GPU: `senpai/gpu_slot.py run --wait ...` is the queue.
- Students post `SENPAI-RESULT` partials with `terminal=false, pending_arms=true`
  after each quick probe so the advisor can steer between arms.
- Cutoff buffer: do not start full evals in the last ~15-20 minutes.

## Potential next directions (round 2 candidates, post round 1 measurements)

- **Scenario D vLLM tuned** — fill the unallocated scenario.
- **FP8 KV cache via FlashInfer** — only if a backend on RTX PRO 6000 actually boots and keeps quality.
- **Speculative decoding** — n-gram speculative for B (long decode), if vLLM speculative-config supports the model on this hardware.
- **TGI** — third engine for portfolio breadth, especially on D.
- **TensorRT-LLM** — high effort; only justifiable late in the run if simpler levers plateau.
- **Cross-scenario confirmation** — once a winner is mature, evaluate across A-D to compute aggregate geomean.
