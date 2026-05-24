# SENPAI Research State

- Updated: 2026-05-24
- Latest direction from human research team: none (no GitHub Issues at boot).
- Active research tag: `ib-20260524-hardened-r3`, advisor branch
  `ib-20260524-hardened-r3`, target base `codex/inferencebench-senpai-target`.
- Hardware: 1x RTX PRO 6000 Blackwell, ~96GB VRAM (shakedown — not
  leaderboard-comparable).

## Current research focus

InferenceBench launcher search for Mistral-7B-Instruct-v0.3 in the 2 hour
SENPAI window. Each PR optimizes ONE scenario at a time and reports the
matching `scenario/<X>/speedup_over_pytorch` against the live PyTorch baseline
captured in scoring assets `rtxpro6000-seed248`.

Three logical students share a single GPU. Coordination via
`senpai/gpu_slot.py`. Heavy launches are serialized; prep work, log analysis,
and quick smoke tests can run in parallel. PR comments are for human-readable
coordination; the slot file is the source of truth for "who is allowed to
run a full eval right now".

## Round 1 hypotheses

Round 1 targets the three highest-leverage scenarios per the public
reference snapshot, with one launcher per student. Each launcher is grounded
in `src/baselines/search_spaces/vllm.yaml` levers and the InferenceBench
contract (foreground OpenAI-compatible server, no daemonization,
`./test_server.sh`/`evaluate.py` flow, full eval + clean relaunch for
terminal results).

| Student      | Scenario | Strategy                                                                                         |
|--------------|----------|--------------------------------------------------------------------------------------------------|
| r3-frieren   | C        | vLLM throughput-first: big batch, prefix cache, chunked prefill, FLASHINFER, FP8 KV cache       |
| r3-fern      | B        | vLLM decode-first: FP8 KV cache + n-gram speculative decoding to cut TPOT                       |
| r3-tanjiro   | A        | vLLM prefill-first: FLASHINFER + chunked prefill tuned for long-input burst c=1                  |

## Potential next research directions (after Round 1)

1. **Scenario D balanced launchers** — once A/B/C have measured baselines,
   try a single Mistral-7B launcher that scores well on D's geomean.
2. **Engine ablations** — re-run the same scenario with SGLang and TGI to
   confirm vLLM is the right base before pursuing custom kernels.
3. **Quantization variants** — beyond FP8 KV cache: weight FP8, AWQ, GPTQ
   on Mistral-7B. Check that MMLU-Pro stays above 0.2831.
4. **Speculative decoding variants** — n-gram vs draft-model vs EAGLE-style
   on Scenario B; tune `num_speculative_tokens` and lookahead.
5. **Attention backend sweep** — FLASH_ATTN vs FLASHINFER vs TRITON_ATTN on
   each scenario with otherwise identical args, to isolate the kernel win.
6. **Chunked prefill / scheduler interplay** — sweep
   `max-num-batched-tokens` alongside `max-num-seqs` for Sc. A and D.
7. **Prefix caching ROI** — measure the prefix-caching delta on Sc. C and D
   under the actual benchmark prompt distribution.
8. **`gpu-memory-utilization` ceiling** — push to 0.92-0.95 with FP8 KV
   cache to expand KV pool without OOM on the 96GB Blackwell.

## Notes

- Reference snapshot is H100; treat it as a ceiling indicator only on RTX
  PRO 6000. We are looking for the right serving recipe, not a numeric
  match.
- Quality gate at `tau=0.95` of PyTorch MMLU-Pro accuracy (0.298) is a hard
  constraint. Any candidate that fails quality is invalid regardless of
  speed.
