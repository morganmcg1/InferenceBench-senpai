# Scenario B — n-gram speculative decoding + FP8 KV — research notes

Working notes for PR #22 (`r1-fern/scenario-b-ngram-spec`, advisor branch `ib-20260523-rerun-r1-advisor`).

## Workload (from scenario.json)

- Model: `mistralai/Mistral-7B-Instruct-v0.3` on 1× NVIDIA RTX PRO 6000 Blackwell, ~96 GB VRAM.
- Profile: `burst`, concurrency 1, 64 requests.
- Per-request: input ~1024 tokens (80–100 % of 1024), output 8192 tokens, temperature 0.7, `ignore_eos=true`.
- Primary metric: `scenario/B/speedup_over_pytorch = (1/tpot.p50_candidate) / (1/tpot.p50_torch)`.
- Quality gate: MMLU-Pro 500-question observed/baseline ≥ 0.95.

## Environment caveats

- Container ships `torch==2.11.0` (CUDA 13.2). Pre-installed `vllm==0.11.0` was built against `torch==2.8.0` + `libcudart.so.12` — fails to import with C++ ABI symbol mismatch (`undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib`). Sourcing `senpai/runtime_env.sh` only resolves the libcudart lookup; it cannot fix the torch ABI.
- vLLM 0.21.0 metadata: `Requires-Dist: torch==2.11.0`. Still exposes the `ngram` speculative method + `prompt_lookup_max/min`. This is the proposed fix; held for advisor approval on PR #22 before installing into shared dist-packages.
- `senpai/preflight.py --scenario B` fails on missing PyTorch baseline + missing MMLU-Pro samples + missing quality registry — same blocker r1-frieren flagged on PR #20.
- `senpai/materialize_requests.py --scenario B` reproduces the tokenizer-roundtrip bug r1-frieren also reported: `_truncate_messages` lands `realized_input_token_count` one token below `min_input_tokens` when target_input_tokens was sampled at the lower edge (820 → 819 on Mistral). `evaluate.py` would hit the same path at runtime unless a `requests.jsonl` is pre-staged.

## Hypothesis decisions (from researcher pass)

The PR assigns n-gram speculative decoding as the experiment. Researcher-agent (Sonnet, 2026-05-23) warns that n-gram is **likely net-negative** for this exact workload because:

- vLLM issue #16258 (vllm-project/vllm, 2025): on a 8192-token decode at γ=5, observed acceptance ≈ 0.698 but system efficiency ≈ 0.426 — speculation overhead dominated, dropping throughput from 504.8 → ~238 tok/s.
- Temperature 0.7 (non-greedy) reduces n-gram acceptance vs. greedy decoding under vLLM's modified rejection sampler.
- `ignore_eos=true` forces a full 8192-token decode, paying speculation overhead on every step. Long generations diverge from the prompt quickly, so prompt-lookup match rate falls.
- The vLLM n-gram docs themselves say spec decode "does not usually yield ITL reductions for all prompt datasets or sampling parameters."

The PR explicitly bans a silent fallback to no-spec — "that is the experiment." I will run the spec decode arm as designed, report the result honestly, and put the negative-result alternative (FP8 KV alone, no chunked prefill at conc=1) in **Suggested follow-ups**.

## Other config decisions

- Keep `--kv-cache-dtype fp8`. Blackwell handles FP8 attention accumulation natively (vLLM April 2026 FP8 KV blog). Risk: Mistral-7B-Instruct-v0.3 is not on the validated MMLU-Pro list in that blog, so I must run the 500-question MMLU-Pro gate before trusting the result.
- Keep `--max-num-batched-tokens 4096`. Concurrency 1 with 1024 prefill fits in a single pass; chunked-prefill is arguably a no-op but matches the PR spec — keeping it for now.
- Keep `--gpu-memory-utilization 0.95`. Reserves ~91.2 GB on a 96 GB card. The previous r3 Scenario B FP8 KV run reported peak ~93 GB (issue #17), so margin is ~1–2 GB; monitor `nvidia-smi`.
- Keep `--max-num-seqs 8`. Concurrency 1 means we never exceed 1 active sequence anyway; the 8-slot ceiling has no cost.
- No EAGLE/Medusa draft model for Mistral-7B-Instruct-v0.3 is available pre-trained, so n-gram is the only zero-training spec-decode option.

## Suggested follow-ups (if n-gram regresses)

1. Re-run with `--no-enable-chunked-prefill` and `--num-speculative-tokens 0` (or drop `--speculative-config` entirely) to isolate FP8 KV + CUDA-graph gain.
2. Try `--speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":3,"prompt_lookup_min":2}'` — lower γ reduces overhead, partially restoring net throughput per issue #16258.
3. Swap to `--quantization fp8` (model weights, not just KV) once quality gate is established. This could halve weight memory traffic per decode step. Run MMLU-Pro again.
4. Sweep `--max-num-batched-tokens` ∈ {2048, 4096, 8192} to confirm the prefill cost is not affecting tpot in a non-obvious way.

## Citations

- vLLM n-gram speculative decoding docs: https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/
- vLLM speculative config schema (vllm 0.21.0): `vllm/config/speculative.py` (`SpeculativeMethod`, `prompt_lookup_max/min`).
- vLLM issue #16258 — γ sweep showing system-efficiency regression.
- vLLM FP8 KV blog (2026-04-22): https://blog.vllm.ai/2026/04/22/fp8-kvcache.html (Blackwell native FP8 attention).
- vLLM EAGLE docs: https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/ (no Mistral-7B-Instruct-v0.3 draft checkpoint).
