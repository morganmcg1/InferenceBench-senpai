# SENPAI Research Results — ib-20260527-lean1-r1

## 2026-05-27 16:46 — PR #128: Scenario C — vLLM FP8 + large batch + prefix caching

- Branch: `tanjiro/scenario-c-throughput-fp8-large-batch`
- Student: tanjiro
- Status: **MERGED — first measured Scenario C row in BASELINE.md**

### Hypothesis

Scenario C is high-load throughput (256 requests across 3 traffic profiles: `burst` c=64, `poisson` 32 r/s, `constant` 16 r/s). Test whether vLLM with `--max-num-seqs=256`, `--max-num-batched-tokens=16384`, `--enable-chunked-prefill`, `--enable-prefix-caching`, `--quantization fp8`, and `--gpu-memory-utilization=0.92` beats the implicit PyTorch baseline.

### Results

| Run | Eval mode | Primary metric | Value | Quality ratio | Quality pass | terminal_eligible |
|---|---|---|---:|---:|---|---|
| `ocz25mgd` | quick | scenario/C/speedup_over_pytorch (geomean) | 2.76x | 0.839 (n=16 screen) | screening-only | False |
| `ckfmuinz` | full | scenario/C/speedup_over_pytorch (geomean) | **25.62x** | **1.0** (n=500) | True | **True** |

- Speed requests: 768/768 succeeded across all 3 profiles, 0 failures.
- Validation: `validation_pass=true`, `baseline_update_allowed=true`.
- VRAM peak: ~89.7 GiB on RTX PRO 6000 (~96 GiB available).
- Raw geomean throughput: 2.17 r/s vs PyTorch baseline 0.0847 r/s.

### Analysis

- The 25.62x speedup is the **first measured Scenario C baseline on RTX PRO 6000** and substantially above the PyTorch baseline.
- Reference (H100 paper, not measured): vLLM default 48.69x. Our 25.62x is below that, consistent with RTX PRO 6000 being slower per-SM than H100 plus different memory hierarchy. This is the new floor on this hardware.
- Quick probe had a misleading cold-start burst (4 requests during CUDA-graph capture inflated TTFT p50 to 21s); steady-state poisson/constant at ~3.86x already foreshadowed the full-eval improvement.
- FP8 weights did not hurt quality at all (ratio 1.0 at n=500). Confirms FP8 weights are a free knob on Mistral-7B-Instruct-v0.3 for this scenario.

### Suggested follow-ups (queued)

- Multi-step scheduler (`--num-scheduler-steps 8`) — assigned as PR #129.
- Compare `--block-size 32` vs default 16 for cache locality at c=64.
- Test launcher without prefix caching to isolate whether LongBench prompts actually share prefixes after templating.
- Lower `--max-num-seqs` to 192 if scheduler thrash dominates burst phase.
