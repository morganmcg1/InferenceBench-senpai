# SENPAI Research State — ib-20260524-ready-r4

- **Snapshot:** 2026-05-24
- **Most recent human-team directive:** none received yet on this launch.
- **Hardware:** RTX PRO 6000 Blackwell (96 GB), one GPU shared by three students
- **Budget:** 2 hour total run
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`

## Current research focus

Initial Round 1 assignments aim to plant a measured launcher on each of the
three highest-headroom scenarios so the live baseline ledger leaves the "none
yet" state quickly. We deliberately split the fleet by scenario rather than by
search-space cell so the early sweep covers the most independent levers:

- **r4-frieren → Scenario C (high-load throughput)** — vLLM continuous batching
  is the reference recipe (46x H100 SMAC3). Test FP8 weights + FP8 KV cache
  with high `--max-num-seqs` and FlashInfer attention on Blackwell.
- **r4-fern → Scenario B (output-heavy / decode)** — concurrency=1 with 8K
  output tokens makes TPOT dominant. Test vLLM with FP8 weights + FP8 KV cache
  + CUDA graphs + n-gram speculative decoding (in-engine, same base model).
- **r4-tanjiro → Scenario A (input-heavy / prefill)** — 8K prefill at
  concurrency=1 makes TTFT dominant. Test vLLM with FlashInfer or FlashAttention
  prefill, large batched-token limit so the full prefill processes in one shot,
  chunked prefill OFF, prefix caching OFF.

Scenario D (balanced) is deferred to Round 2 — likely best served by lifting
the strongest single-scenario launcher and confirming geomean balance.

## Hardware finding (2026-05-24 07:40 UTC, from r4-frieren PR #46 commit 20f08ea)

Critical platform constraint discovered during Round 1 cold-start: on this
RTX PRO 6000 pod (CC 12.0 Blackwell, CUDA 13.2 nvcc, vLLM 0.11):

- **FlashInfer JIT fails to build kernels for sm_120.** `VLLM_ATTENTION_BACKEND=FLASHINFER`
  is not usable on this hardware/toolchain combination.
- **FlashAttention v3 FP8-KV path is Hopper-only (CC 9.x).** Combining
  `VLLM_ATTENTION_BACKEND=FLASH_ATTN` with `--kv-cache-dtype fp8` errors on
  Blackwell.
- **TRITON_ATTN is the working backend** for FP8 W8A8 + FP8 KV on this pod.
- The full env override block frieren is using:
  - `VLLM_ATTENTION_BACKEND=TRITON_ATTN`
  - `VLLM_USE_FLASHINFER_SAMPLER=0`
  - `VLLM_DISABLE_FLASHINFER_PREFILL=1`

This finding was propagated to PRs #47 and #48 so fern and tanjiro can update
their launchers before they claim the GPU.

## Potential next research directions

Once Round 1 launcher recipes return, consider in priority order:

1. **Compose winners.** If Scenario C/B/A converge on overlapping flag sets
   (e.g. FP8 KV + FlashInfer), promote that combo to a Scenario D probe.
2. **Speculative decoding diversification.** If Scenario B improves with
   n-gram speculation, try larger draft windows or compare against EAGLE-style
   draft models that preserve the assigned base model architecture.
3. **Engine bake-off.** SGLang vs vLLM head-to-head on the winning scenario,
   especially for C where SGLang default already shows 51x request throughput
   on H100 reference.
4. **Kernel tuning.** Compare `FLASH_ATTN` vs `FLASHINFER` vs `TRITON_ATTN`
   on the best launcher per scenario; Blackwell may prefer different paths
   than the H100 reference numbers.
5. **TGI quantization variants** (FP8, eetq, GPTQ) if vLLM/SGLang plateau.
6. **TensorRT-LLM** if there is time after a vLLM/SGLang frontier is set.
7. **Aggregate confirmation.** Once any scenario has a strong launcher,
   schedule a cross-scenario A-D confirmation run.

## Operating constraints

- One GPU, three students. Heavy GPU work must be serialized via PR comments.
  Use `SLOT-FREE` markers on PR comments when a server is torn down so the next
  student can claim the GPU. No broad `pkill` — students kill only their own
  server process group.
- Every scenario PR must use the official `evaluate.py` against the precomputed
  request files in `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`.
- Quality gate (MMLU-Pro greedy, tau=0.95 of PyTorch baseline) must pass before
  any result is treated as terminal/mergeable.
- Reject `SENPAI-RESULT` payloads where `primary_metric.value` is a raw
  objective rather than the `speedup_over_pytorch` ratio.
