# SENPAI Research State
- 2026-05-28 16:42 UTC
- No messages from human researcher team yet.

## Current research focus
Scenario B (output-heavy decode, `scenario/B/speedup_over_pytorch`).  
One RTX PRO 6000 GPU shared between two logical students; 2-hour run window
closing at 2026-05-28T18:36:57Z.

PyTorch reference: TPOT p50 = 0.02515 s/token → raw objective = 39.76 tok/s.  
Public H100 best for Scenario B: 15.23x (SMAC3 search, not yet matched on RTX PRO 6000).

## Active assignments

| Student | PR | Hypothesis slug | Primary lever |
|---|---|---|---|
| scen-b-frieren | #159 | scenario-b-ngram-spec-decode | N-gram speculative decoding (k=3,5) + CUDA graphs |
| scen-b-fern | #162 | scenario-b-fp8-weight-scheduler | FP8 weight quantization + minimal scheduler for c=1 |

## Research themes

1. **Decode-step amplification** (frieren): n-gram spec decode emits >1 token/step at
   concurrency 1. At burst c=1 and 8192 output tokens, accepted proposals compound
   directly into TPOT speedup. k=5 vs k=3 comparison measures token-acceptance
   quality vs safety on this workload.

2. **Bandwidth reduction** (fern): FP8 weight quantization halves weight-load bandwidth,
   directly cuts per-token memory cost. Stacking with minimal-scheduler settings removes
   batching overhead that is irrelevant at c=1. Expected to produce clean, stackable wins.

## Potential next research directions (post round-1)

- Stack winning levers from both arms (spec decode + FP8 weights + lean scheduler)
- Try TRITON_ATTN attention backend vs FLASH_ATTN on this Blackwell GPU
- Attempt torch.compile (piecewise CUDA graphs) for additional kernel-launch savings
- SGLang as alternative engine (has good RadixAttention, less RTX PRO 6000 hardening history)
- Larger block_size=64 experiment if block_size=32 shows benefit
- `--max-num-seqs 2` probe: does accepting one extra in-flight request at c=1 pipeline better?
- FP8 KV cache (blocked on FLASH_ATTN incompatibility; needs FLASHINFER enabled or TRITON backend)

## Current baseline
No terminal result yet. First launch.
