# BASELINE — ib-20260524-leasefix-r3

- **Date opened:** 2026-05-24
- **Hardware:** NVIDIA RTX PRO 6000 Blackwell (97887 MiB) — SHAKEDOWN MODE
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Dataset seed:** 248
- **Quality samples:** `mmlu_pro` 500 questions, seed 248
- **Preflight:** PASS (imported from `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`)
- **W&B project:** `wandb-applied-ai-team/inferencebench-senpai`

> NOTE: RTX PRO 6000 shakedown — NOT leaderboard-comparable. The 2026-05-21
> reference snapshot in `program.md` is for H100. Speedups here may differ
> meaningfully from H100 speedups and must be repeated on H100 before claiming
> leaderboard parity.

## PyTorch Baselines (raw, RTX PRO 6000)

| Scenario | Profile | ttft.p50 (s) | tpot.p50 (s) | req/s | gen tok/s |
|---|---|---:|---:|---:|---:|
| A (input-heavy) | burst | 0.4385 | 0.0258 | 0.0709 | 35.7 |
| B (output-heavy) | burst | 0.0709 | 0.0252 | 0.0133 | 39.2 |
| C (high-load) | burst | 0.0702 | 0.0237 | 0.0845 | 39.2 |
| C (high-load) | poisson | 0.0702 | 0.0240 | 0.0847 | 38.4 |
| C (high-load) | constant | 0.0704 | 0.0241 | 0.0849 | 38.6 |
| D (general) | burst | 0.2123 | 0.0251 | 0.0382 | 38.2 |

## Quality Gate

- Backend: torch, model: mistralai/Mistral-7B-Instruct-v0.3
- Baseline mmlu_pro observed accuracy: 0.298 (n=500, seed=248)
- Pass requires `observed >= 0.95 * 0.298 ≈ 0.283`

## Current Best Launcher (per scenario)

| Scenario | Launcher path | Engine | Speedup | W&B run | PR |
|---|---|---|---:|---|---|
| A | `src/starting_points/vllm_running/start_server.sh` | vLLM default | 1.00x (PyTorch baseline placeholder) | — | — |
| B | `senpai/launchers/scenario_b/vllm-ngram5-fp8kv/start_server.sh` | vLLM (n-gram spec-decode num_spec=5, FLASH_ATTN, max_num_seqs=8, no chunked-prefill / no prefix cache, block_size=16) | **2.694x** | `rnc0c1by` | #88 |
| C | `src/starting_points/vllm_running/start_server.sh` | vLLM default | 1.00x (PyTorch baseline placeholder) | — | — |
| D | `senpai/launchers/D/vllm-balanced-burst/start_server.sh` | vLLM (chunked-prefill + CUDA graphs + prefix cache, max_num_seqs=16, max_num_batched_tokens=8192, gpu_mem=0.92, FLASH_ATTN) | **1.305x** | `z1m0zvsl` | #77 |

The advisor-tracked current best for each scenario is the speedup measured by
the SENPAI evaluator on this RTX PRO 6000 pod, not the H100 README references.
The default `vllm_running` launcher is the baseline placeholder until a first
SENPAI vLLM measurement supersedes it; treat any terminal SENPAI result that
clears the quality gate as the new current best for that scenario.

## Update History

- 2026-05-24 — Advisor opens ledger. Preflight PASS, no SENPAI measurements yet.
- 2026-05-24 22:55 — **PR #77 merged.** Scenario D: vLLM balanced-burst launcher (chunked-prefill + CUDA graphs + prefix cache, max_num_seqs=16, max_num_batched_tokens=8192, gpu_mem=0.92, FLASH_ATTN) reached **1.305x speedup over PyTorch**, raw obj 2.519 vs 1.930. Quality PASS (mmlu_pro 0.308 vs 0.298 baseline, ratio 1.034). TTFT.p50 0.1735 (-18.3%), TPOT.p50 0.01697 (-32.3%), req/s 0.04706 (+23.1%), gen tok/s 56.24 (+47.2%), 96/96 success. W&B run `z1m0zvsl`. Relaunch eval not captured due to a wrapper PGID bug (separate tooling fix); launcher is deterministic env-only, first eval is already the fresh-launch case.
- 2026-05-24 23:43 — **PR #88 merged.** Scenario B: vLLM n-gram speculative decoding (num_speculative_tokens=5, prompt_lookup_min=2, prompt_lookup_max=4) with FLASH_ATTN, max_num_seqs=8, block_size=16, no chunked-prefill, no prefix cache, gpu_mem=0.90, FP8 KV cache dropped (FA backend rejects FP8 KV on Blackwell). **2.694x speedup on B**, raw obj 107.09 vs 39.76 (1/tpot.p50). Quality PASS (mmlu_pro 0.302 vs 0.298 baseline, ratio 1.013). TTFT.p50 0.0641 (-9.6%), TPOT.p50 0.00934 (-62.9%), ITL.p50 0.01309, gen tok/s 103.68 (+165%), req/s 0.02187 (+64%), 64/64 success, VRAM peak 87.7 GB. W&B run `rnc0c1by`. The winning launcher is at `senpai/launchers/scenario_b/vllm-ngram5-fp8kv/start_server.sh`; the parameterized template at `senpai/launchers/B/vllm-ngram-spec/start_server.sh` is documentation only.
