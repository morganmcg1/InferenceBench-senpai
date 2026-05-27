# SENPAI Research State — ib-20260527-latest3-r1

- **As of:** 2026-05-27 16:22 UTC (107 min into 2h run — END OF LAUNCH)
- **Last human directive:** none (no open GitHub Issues from research team)

## Research focus

InferenceBench LLM inference optimization. Goal: maximize per-scenario speedup
over PyTorch baseline on Mistral-7B-Instruct-v0.3 (RTX PRO 6000 shakedown, single
GPU, 2 hour budget) while passing the MMLU-Pro quality gate.

## Final launch state

Two merged terminal winners:

| Scenario | Result | Launcher | PR |
|---|---|---|---|
| **B (output-heavy)** | **2.750x** — vLLM ngram-spec (num_spec=5) | `senpai/launchers/B/vllm-ngram-spec/start_server.sh` | #122 merged |
| **C (high-load)** | **22.48x** — SGLang mem085 mrr128 | `senpai/launchers/C/sglang-mem085-mrr128/start_server.sh` | #124 merged |
| A (input-heavy) | 1.273x quick only (non-mergeable) | `senpai/launchers/A/vllm-chunked-8k/start_server.sh` | #123 closed |
| D (general) | No eval — launcher committed | `senpai/launchers/D/vllm-d-ngram5/start_server.sh` | #125 closed |

Aggregate of merged winners (B and C only): geomean(2.750, 22.48) = **7.87x** (not final — A and D outstanding).

## Key discoveries this launch

- **ngram speculative decoding is the dominant lever for output-heavy workloads on RTX PRO 6000.** vLLM with `num_speculative_tokens=5, prompt_lookup_min=2, max=4` pushed B from 1.44x (CUDA graphs alone) to 2.75x full. Quick showed 3.44x — full eval moderated this to 2.75x.
- **SGLang throughput engine is the dominant lever for high-concurrency workloads.** mem-fraction-static 0.85 + max-running-requests 128 gave 22.48x on C vs 1.96x vLLM default. Critically, quick mode (4 req/profile) wildly underestimates C speedup — 4.01x quick vs 22.48x full.
- **FP8/FlashInfer unavailable on RTX PRO 6000** (SM120 Blackwell): runtime_env.sh forces both off. Scenario A headroom is capped at ~1.3x without these. H100 is needed for A gains.
- **B VRAM footprint concern**: frieren's winner uses 87.7 GB VRAM at `--gpu-memory-utilization 0.90`. Needs `--gpu-memory-utilization 0.80` to fit H100 80GB.

## Highest-priority next directions (next launch)

1. **Scenario D full eval** — tanjiro's ngram-spec launcher committed at `senpai/launchers/D/vllm-d-ngram5/start_server.sh`. No implementation needed; run quick then full. Expected: ≥2x based on B transfer. Full D ~26 min.
2. **Scenario A full eval** — fern's chunked-prefill launcher committed at `senpai/launchers/A/vllm-chunked-8k/start_server.sh`. No implementation needed; run full A (~73 min). Expected: ~1.3-1.5x (unless FP8/FlashInfer can be enabled).
3. **Scenario A: SGLang RadixAttention probe** — different prefill path that may bypass the FlashInfer block. Could provide better TTFT than vLLM chunked-prefill on RTX PRO 6000.
4. **Push B further**: frieren suggested EAGLE/MTP draft model speculation (vs ngram) for 4-6x+ decode; 4-bit AWQ/GPTQ quantization as orthogonal lever. Either could push B toward the SMAC3 15.23x reference.
5. **B H100 VRAM fix**: `--gpu-memory-utilization 0.80` variant of the ngram-spec launcher for leaderboard-comparable H100 runs.
6. **Push C further**: current 22.48x on RTX PRO 6000 vs H100 reference 51.12x SGLang. Try max-running-requests 256 + LPM scheduler + torch compile on next hardware.
7. **Scenario D: SGLang probe** — not tested this run; portfolio diversification. SGLang's throughput advantage may also apply to D's balanced workload.
8. **Cross-scenario aggregate** — once A and D have terminal winners, compute geomean(A,B,C,D) for leaderboard submission readiness.
