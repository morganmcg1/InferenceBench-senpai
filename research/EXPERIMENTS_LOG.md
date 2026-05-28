# SENPAI Research Results

## 2026-05-28 11:53 UTC — PR #137: Scenario A vLLM prefill TTFT arms (one-shot vs chunked vs FP8 weights)

- **Branch:** `frieren/sc-a-vllm-prefill-ttft-arms`
- **Student:** frieren
- **Hypothesis:** On Scenario A (8192-token input-heavy, concurrency 1 burst), reducing
  weight-read bandwidth via FP8 weight quantization should cut TTFT proportionally, because
  TTFT is dominated by the prefill matmul chain which is memory-bandwidth-bound on Mistral-7B at
  concurrency 1.

**Quick-probe results:**

| Arm | Config | TTFT.p50 (s) | Speedup (quick) | Notes |
|---|---|---:|---:|---|
| arm1 one-shot BF16 | `--no-enable-chunked-prefill`, `--max-num-batched-tokens 10240` | 0.3435 | 1.276x | vLLM 0.11 forces chunked prefill anyway when batched_tokens < model_len |
| arm2 chunked BF16 | `--enable-chunked-prefill`, `--max-num-batched-tokens 10240` | 0.3427 | 1.279x | Identical to arm1 — same regime, within noise |
| **arm3 FP8 weights** | arm1 + `--quantization fp8` | **0.2307** | **1.901x** | Clean boot on Blackwell; quality n=16 unchanged vs BF16 |

**W&B quick runs:** 5iump62l (arm1), r20z2rm0 (arm2), eyvyj8oz (arm3)

**Full eval result (arm3 FP8):**

| Metric | Value |
|---|---|
| `scenario/A/speedup_over_pytorch` | **1.8664** |
| `scenario/A/inverse_ttft_p50` | 4.256 (PyTorch 2.280) |
| TTFT.p50 / p90 / p99 | 0.2349 / 0.2613 / 0.2681 s |
| TPOT.p50 | 0.0175 s |
| Request throughput | 0.0903 req/s |
| MMLU-Pro accuracy | 0.288 (baseline 0.298, ratio 0.966, n=500) |
| Quality gate | **PASS** (ratio 0.966 ≥ tau 0.95) |
| Speed success | 128/128 (failure_rate 0.0) |
| VRAM peak | 93267 / 97887 MiB |
| W&B run | **izg22lch** |

**Merged:** Yes — squash-merged to `ib-20260528-12h-r2` at 11:53 UTC.

**Analysis and conclusions:**

- FP8 weight-only quantization (`--quantization fp8`) gives a clean 1.87x TTFT speedup on
  Scenario A versus the PyTorch baseline. This is roughly the expected theoretical gain for
  halving weight-read bandwidth on a prefill-dominated workload at concurrency 1.
- The "one-shot vs chunked prefill" distinction is moot under vLLM 0.11: the engine forces
  chunked prefill whenever `max_num_batched_tokens < max_model_len`, so both arms ran with
  a single 10240-token chunk (larger than the 8192-token input), behaving identically.
- Quality at n=500 shows only ~1 pp absolute accuracy drop (0.298 → 0.288), comfortably
  above the 0.95 × baseline gate. FP8 quantization on Mistral-7B-Instruct-v0.3 is safe.
- The H100 public reference for Scenario A is 4.37x (SMAC3). Our 1.87x confirms FP8 alone
  is not sufficient to reach that level; further levers (larger batched-token tuning, chunked
  prefill with overlap at higher concurrency, attention backend, FP4/W4A8) are required.
- **Bug identified:** `senpai/create_task_workspace.py` generates `eval_env.sh` with a
  relative `PROBLEM_DIR`, which fails to source `runtime_env.sh`, leaving FlashInfer enabled
  on Blackwell — causing a JIT kernel compile failure on SM120f. Workaround: `export
  PROBLEM_DIR="$(realpath $PROBLEM_DIR)"` before eval. Should be fixed in the script.

**Next experiment:** frieren assigned PR #139 (Scenario D, vLLM FP8 + n-gram speculative
composition — testing whether both Round-1 winning levers stack on the balanced scenario).
