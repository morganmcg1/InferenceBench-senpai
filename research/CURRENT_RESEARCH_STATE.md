# SENPAI Research State

- **Updated:** 2026-05-28T18:08Z
- **Most recent direction from human researcher team:** No GitHub Issues. Run
  scope fixed by operator: Scenario A parity launch, RTX PRO 6000 shakedown,
  2h budget, 1 GPU shared, 2 logical students. Launch closes
  `2026-05-28T18:36:57Z`.

## Terminal winner (merged, final result for this launch)

**PR #160 — scen-a-fern/fern-fp8-weights — 1.8873x speedup — MERGED to `ib-20260528-scen-a-r1`**

vLLM `--quantization fp8` on-the-fly weight quantization. Full eval 128/128
success, MMLU-Pro ratio 0.9597 (pass, tight but valid), TTFT.p50 0.2325s, W&B
`bt2ckkzz`. BASELINE.md updated.

## Probes resolved this round

| PR | Student | Arm | Outcome | Final reading |
|---|---|---|---|---|
| #158 | scen-a-frieren | A0/A1 BF16 prefill tuning | Closed | BF16 ceiling ~1.276x; prefill structure is not a lever at burst concurrency 1 |
| #160 | scen-a-fern | B0/B1 FP8 weights + full eval | **MERGED** | 1.8873x winner; B1 tight prefill structure no-op over B0 |
| #168 | scen-a-fern | C0 FlashInfer + FP8 | Closed | failed_to_boot — `_sm_scale` AssertionError in vLLM FlashInfer wrapper |
| #169 | scen-a-frieren | D0 FP8 + enforce_eager | Closed | -4.1% TTFT vs winner; cudagraphs are net-positive at burst concurrency 1 |
| #171 | scen-a-fern | E0 FlashInfer + FP8 + enforce_eager (rescue) | Closed | Same `_sm_scale` assertion with cudagraphs off — wrapper-construction path itself is broken |

## Key findings (carry forward)

- **FP8 weights are the dominant Scenario A lever** on RTX PRO 6000 in this
  image. Going from BF16 (1.276x) to FP8 weight GEMM (1.887x) is a ~48%
  relative TTFT reduction.
- **Prefill structure tuning** (chunked-prefill toggle, max-model-len,
  max-num-batched-tokens) **does not move TTFT** at concurrency 1 with
  8192-token inputs — confirmed in both BF16 (PR #158 A0/A1) and FP8
  (PR #160 B0 vs B1). The bottleneck is the GEMM, not scheduling.
- **CUDA graphs are net-positive** at burst concurrency 1 with FP8 weights
  on this hardware (PR #169 D0: -4.1% TTFT when disabled). Future arms must
  keep cudagraphs enabled.
- **FlashInfer prefill is unreachable on vLLM 0.11.0 + this RTX PRO 6000
  image.** Both PR #168 (cudagraph on) and PR #171 (cudagraph off) hit
  `vllm/v1/attention/backends/flashinfer.py:972 assert decode_wrapper._sm_scale
  == self.scale` during `kernel_warmup`. The wrapper-construction path itself
  is broken; `--enforce-eager` is not a workaround.
- **FlashInfer sampler JIT** fails on this image (missing `curand.h`).
  Workaround: explicit `VLLM_USE_FLASHINFER_SAMPLER=0` and
  `VLLM_DISABLE_FLASHINFER_SAMPLING=1` inside every launcher. The
  `senpai/runtime_env.sh` defaults do not persist through the gpu_slot shell.

## Infrastructure carry-overs for next launch

- **cublas link bug, cold FlashInfer JIT cache** (Frieren PR #169):
  `Fp8LinearOp` on SM_120 routes to `flashinfer_w8a8_scaled_mm` → JIT compile
  of `gemm.so` → link fails (`-lcublas`, `-lcublasLt`) because nvidia pip
  cublas ships only `libcublas.so.12`/`libcublasLt.so.12`. The PR #160 winner
  launcher and `runtime_env.sh` will crash on any fresh JIT cache. Frieren's
  workaround (symlink to unversioned `.so` + prepend `LIBRARY_PATH`) is
  embedded in `senpai/launchers/A/frieren-eager-fp8/arm_d0_fp8_eager.sh`.
  Next-launch task: patch `senpai/runtime_env.sh` to apply the same fix
  upstream of every launcher.

## Time budget (as of 18:08Z)

- Cutoff: `2026-05-28T18:36:57Z` → **~29 min remaining**.
- Reserve 10-15 min for final review/baseline updates → effective student
  probe window ends ~18:22-18:27.
- A single new quick probe could fit if started immediately, but the
  marginal value over the documented winner is low and the carry-over
  research signal already exists for next-round work.

## Next-round directions (priority-ordered)

1. **Patch `senpai/runtime_env.sh`** to bake the cublas symlink + `LIBRARY_PATH`
   fix in so FP8 launchers survive a cold FlashInfer JIT cache. This is an
   infra bug-fix PR, separate from any hypothesis arm. High priority — the
   current winner launcher is fragile to fresh-pod environments.
2. **vLLM V0 engine + FP8** (`VLLM_USE_V1=0`): pure engine-version comparison
   on top of the FP8 winner. Cleanest single-flag arm not yet tested.
3. **Triton attention backend + FP8** (`VLLM_ATTENTION_BACKEND=TRITON_ATTN`):
   alt prefill path that bypasses both FlashAttention-2 and FlashInfer's
   broken wrapper. Worth a quick probe.
4. **`torch.compile` optimization level + FP8**: vLLM 0.11 supports compile
   modes that may further fuse prefill ops. Quick to probe.
5. **GPTQ or AWQ instead of naive FP8**: a quantization scheme with better
   quality preservation than naive FP8 weight-only could both pass MMLU-Pro
   more safely (current ratio 0.9597 is only 0.0097 above the τ=0.95 floor)
   and may maintain or improve speed.
6. **SGLang Triton backend + FP8**: engine-family diversity. The natural
   path if vLLM hits a hard ceiling we cannot move past.
7. **vLLM patch to relax the `_sm_scale` assertion** or vLLM version bump
   that fixes FlashInfer 0.4.x on SM_120 — both unlock the FlashInfer prefill
   direction that this launch confirmed is otherwise closed on this image.
8. **H100 leaderboard run**: repeat the winning FP8 recipe on H100 for
   leaderboard-comparable claim. The RTX PRO 6000 numbers here are shakedown
   only.

## Final probe this round

**PR #176** — Frieren arm F0: `VLLM_USE_V1=0` + FP8 weights (V0 vs V1 engine
comparison). Single env-var toggle on top of the FP8 winner recipe, with
Frieren's cublas symlink workaround embedded. Assigned 18:12Z; expected result
~18:20-18:26Z. Quick-only; terminal=false regardless of result. If V0 is faster,
flags as high-priority next-round full-eval candidate.

Fern remains idle for the final ~25 min — not enough wall time for a second
independent probe given shared-GPU queuing.
