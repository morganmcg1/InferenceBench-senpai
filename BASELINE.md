# SENPAI InferenceBench Baseline Ledger

- **Advisor branch:** `ib-20260528-12h-r2`
- **Research tag:** `ib-20260528-12h-r2`
- **Hardware (current):** NVIDIA RTX PRO 6000 (~96 GB VRAM) — shakedown only,
  not leaderboard-comparable to H100 reference snapshot.
- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **W&B:** `wandb-applied-ai-team/inferencebench-senpai`
- **Scoring assets:** imported from
  `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`
  (seed 248, all four scenarios). Preflight: PASS as of 2026-05-28 10:42 UTC.

Reference snapshot for the H100 80GB / 2-hour-per-run leaderboard setting
(2026-05-21, copied from `program.md` for orientation only — not RTX PRO 6000
baselines):

| Method | Sc. A TTFT | Sc. B TPOT | Sc. C req/s | Sc. D geomean |
|---|---:|---:|---:|---:|
| PyTorch baseline | 1.00x | 1.00x | 1.00x | 1.00x |
| vLLM default | 1.25x | 2.25x | 48.69x | 1.96x |
| SGLang default | 1.22x | 1.77x | 51.12x | 2.14x |
| HF TGI default | 1.14x | 1.37x | 41.94x | 1.80x |
| Random search vLLM 2h | 4.21x | 11.34x | 41.81x | 5.42x |
| TPE search vLLM 2h | 4.48x | 14.76x | 43.46x | 5.58x |
| SMAC3 search vLLM 2h | 4.37x | 15.23x | 46.70x | 5.69x |
| Sonnet 4.6 agent | 3.47x | 12.03x | 33.93x | 3.01x |

## Current best valid launcher (RTX PRO 6000 shakedown)

| Scenario | Primary metric | Best speedup vs PyTorch | Launcher | W&B run | PR |
|---|---|---:|---|---|---|
| **A** | scenario/A/speedup_over_pytorch | **1.866x** | `senpai/launchers/A/frieren-vllm-ttft/arm3_fp8_weights.sh` | izg22lch | #137 |
| **B** | scenario/B/speedup_over_pytorch | **2.687x** | `senpai/launchers/B/fern-vllm-tpot/arm3_ngram_spec.sh` | dav3txgq | #136 |
| **C** | scenario/C/speedup_over_pytorch | **21.052x** | `senpai/launchers/C/fern-vllm-throughput/arm1_bf16_highconc.sh` | tebmnnza | #140 |
| **D** | scenario/D/speedup_over_pytorch | **2.073x** | `senpai/launchers/D/frieren-vllm-composition/arm3_fp8_and_ngram.sh` | 40f15iox | #139 |

### Scenario D — current winner (PR #139, merged 2026-05-28) — supersedes PR #138

- **Engine:** vLLM 0.11.0, FlashAttention backend, FP8 weight quantization + n-gram speculative decoding, BF16 KV cache
- **Key flags:** `--max-num-seqs 32 --max-num-batched-tokens 4096 --enable-chunked-prefill --no-enable-prefix-caching --kv-cache-dtype auto --quantization fp8 --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'`
- **TTFT.p50:** 0.1161 s (PyTorch 0.2123 s) — FP8 cuts prefill
- **TPOT.p50:** 0.0104 s (PyTorch 0.0251 s) — n-gram spec cuts decode
- **req/s:** 0.0776 (PyTorch 0.0382)
- **Geomean (1/ttft.p50, 1/tpot.p50, req/s):** 4.001 (PyTorch 1.930)
- **Speedup:** 2.073x (vs prior Sc D best of 1.247x SGLang, +66%)
- **Quality:** MMLU-Pro 0.284 obs / 0.298 baseline = ratio 0.953 (gate 0.95, n=500) ✓
- **Speed success:** 96/96 (failure_rate 0.0) ✓
- **VRAM peak:** 91.7 GiB / 97.9 GiB
- **W&B run:** 40f15iox
- **Reproduce (from task workspace):** `cp senpai/launchers/D/frieren-vllm-composition/arm3_fp8_and_ngram.sh ./start_server.sh && python evaluate.py --json-output-file metrics_full.json`
- **Key insight:** FP8 + n-gram compose additively: FP8 attacks the 4096-token prefill stage (TTFT -45%), n-gram attacks the 2048-token decode stage (TPOT -59%). Both levers target independent bottlenecks in balanced Sc D, producing clean multiplicative lift.

### Scenario C — current winner (PR #140, merged 2026-05-28)

- **Engine:** vLLM 0.11.0, FlashAttention backend, BF16 weights, BF16 KV cache (no FP8, no prefix caching)
- **Key flags:** `--max-num-seqs 64 --max-num-batched-tokens 8192 --enable-chunked-prefill --no-enable-prefix-caching --kv-cache-dtype auto --gpu-memory-utilization 0.92`
- **Geomean req/s:** 1.783 (PyTorch 0.0847) across burst/poisson/constant profiles
- **Per-profile req/s:** burst 2.605, poisson 1.854, constant 1.175 (all 256/256 ✓)
- **Speedup:** 21.052x
- **Quality:** MMLU-Pro 0.298 obs / 0.298 baseline = ratio 1.000 (gate 0.95, n=500) ✓
- **Speed success:** 768/768 (failure_rate 0.0) ✓
- **VRAM peak:** 90815 MiB / 97887 MiB
- **W&B run:** tebmnnza
- **Reproduce (from task workspace):** `cp senpai/launchers/C/fern-vllm-throughput/arm1_bf16_highconc.sh ./start_server.sh && python evaluate.py --json-output-file metrics_full.json`
- **Key insight:** Sc C is KV-cache-memory-bound and scheduler-overhead-bound (not weight-read-bound). FP8 dequant overhead *hurts* (-3.3% vs BF16); prefix caching gives +1% noise. Setting `--max-num-seqs 64` matched to burst concurrency is the key lever; PyTorch at 0.0847 req/s is serialized; vLLM batching 64 concurrent streams achieves 2.60 req/s in burst.

### Scenario D — prior winner (PR #138, merged 2026-05-28, superseded by PR #139)

- **Engine:** SGLang (per-PR venv, `sglang[all]`), Triton attention backend, BF16 KV cache
- **Relaunch contract:** `libnuma.so.1` bundled under `senpai/launchers/D/tanjiro-sglang/lib/`; venv auto-bootstrapped via `uv venv` + `pip install sglang[all]` if `/tmp/inferencebench-engine-venvs/sglang-pr-138` absent
- **Key flags:** `--context-length 16384 --mem-fraction-static 0.80 --attention-backend triton --trust-remote-code`
- **Geomean (1/ttft.p50, 1/tpot.p50, req/s):** 2.413 (PyTorch 1.930)
- **Speedup:** 1.247x
- **Quality:** MMLU-Pro 0.310 obs / 0.298 baseline = ratio 1.040 (gate 0.95, n=500) ✓
- **Speed success:** 96/96 (failure_rate 0.0) ✓
- **W&B run:** nf10i0y2
- **Reproduce (from task workspace):** `cp senpai/launchers/D/tanjiro-sglang/arm1_sglang_default.sh ./start_server.sh && SGLANG_LIB_DIR="$(realpath senpai/launchers/D/tanjiro-sglang/lib)" python evaluate.py --json-output-file metrics_full.json`

### Scenario B — current winner (PR #136, merged 2026-05-28)

- **Engine:** vLLM 0.11.0, FlashAttention backend, n-gram (prompt-lookup) speculative decoding, BF16 KV cache
- **Key flags:** `--speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}' --max-num-seqs 16 --max-num-batched-tokens 2048 --enable-chunked-prefill --no-enable-prefix-caching --gpu-memory-utilization 0.92`
- **TPOT.p50:** 0.00936 s (PyTorch 0.0252 s) — exact verification, zero quality risk from speculation
- **Speedup:** 2.687x (inverse_tpot_p50: 106.84 vs 39.76 tok/s)
- **Quality:** MMLU-Pro 0.300 obs / 0.298 baseline = ratio 1.007 (gate 0.95, n=500) ✓
- **Speed success:** 64/64 (failure_rate 0.0) ✓
- **VRAM peak:** 90305 MiB / 97887 MiB
- **Reproduce (from task workspace):** `cp senpai/launchers/B/fern-vllm-tpot/arm3_ngram_spec.sh ./start_server.sh && python evaluate.py --json-output-file metrics_full.json`

### Scenario A — current winner (PR #137, merged 2026-05-28)

- **Engine:** vLLM 0.11.0, FlashAttention backend, FP8 weight-only quantization, BF16 KV cache
- **Key flags:** `--quantization fp8 --max-num-seqs 8 --max-num-batched-tokens 10240 --no-enable-chunked-prefill --no-enable-prefix-caching --gpu-memory-utilization 0.92`
- **TTFT.p50:** 0.2349 s (PyTorch 0.4385 s)
- **Speedup:** 1.866x (inverse_ttft_p50: 4.256 vs 2.280)
- **Quality:** MMLU-Pro 0.288 obs / 0.298 baseline = ratio 0.966 (gate 0.95, n=500) ✓
- **Speed success:** 128/128 (failure_rate 0.0) ✓
- **VRAM peak:** 93267 MiB / 97887 MiB
- **Reproduce (from task workspace):** `cp senpai/launchers/A/frieren-vllm-ttft/arm3_fp8_weights.sh ./start_server.sh && python evaluate.py --json-output-file metrics_full.json`

PyTorch baseline raw objectives (from imported metrics):

| Scenario | Profile | TTFT.p50 | TPOT.p50 | Throughput (req/s) |
|---|---|---:|---:|---:|
| A | burst | 0.4385s | 0.0258s | 0.0709 |
| B | burst | 0.0709s | 0.0252s | 0.0133 |
| C | burst | 0.0702s | 0.0237s | 0.0845 |
| C | poisson | 0.0702s | 0.0240s | 0.0847 |
| C | constant | 0.0704s | 0.0241s | 0.0849 |
| D | burst | 0.2123s | 0.0251s | 0.0382 |

## Provisional / quick-only results

These are `--quick` runs (`request_limit=4`, `quality_n=16`). Quality ratios at
this sample size are statistical noise, not gate-eligible. Speedups are
research signal only and must not be used to update the current-best row.

| Scenario | Arm | Quick speedup | Quality ratio (n=16) | W&B run | PR | Notes |
|---|---|---:|---:|---|---:|---|
| B | fern arm1 BF16 tuned | 1.438x | 0.839 | ex8p5t6j | #136 | vLLM 0.11, max_num_seqs=16, chunked prefill on |
| B | fern arm2 FP8 weights | 1.474x | 0.839 | ycwnvt87 | #136 | +`--quantization fp8`; within 5% of arm1 |
| B | fern arm3 n-gram spec | **3.518x** | 0.839 | du9y0jz8 | #136 | +n-gram spec, num_speculative_tokens=5 — promoted to full eval |
| A | frieren arm1 one-shot prefill | 1.276x | 0.839 | 5iump62l | #137 | vLLM 0.11, no chunked prefill flag (vLLM forces it anyway) |
| A | frieren arm2 chunked prefill large | 1.279x | 0.839 | r20z2rm0 | #137 | within noise of arm1 |
| A | frieren arm3 FP8 weights | **1.901x** | 0.839 | eyvyj8oz | #137 | +`--quantization fp8` — promoted to full eval |
| D | tanjiro arm1 sglang_default | 1.167x | 0.629 | 2nfds9ud | #138 | SGLang per-PR venv; initial quick (libnuma not yet bundled) |
| D | tanjiro arm2 sglang_tuned | 1.183x | 0.629 | 8zul6lqz | #138 | +chunked-prefill-size=4096, lpm scheduler; within 5% of arm1 |
| D | frieren arm1 fp8 only | 1.326x | 0.839 | — | #139 | vLLM FP8-only quick probe |
| D | frieren arm2 ngram only | 1.697x | 0.839 | — | #139 | vLLM n-gram-only quick probe |
| D | frieren arm3 fp8+ngram | **1.810x** | 0.629 | — | #139 | vLLM FP8+n-gram composition — promoted to full eval |
| C | fern arm1 BF16 high-conc | **3.888x** | 0.839 | qvcrnc4s | #140 | vLLM, max_num_seqs=64, batched_tokens=8192 — promoted to full eval |
| C | fern arm2 + prefix cache | 3.928x | 0.839 | 2rtv6gxb | #140 | +`--enable-prefix-caching`; within 1% of arm1 |
| C | fern arm3 + FP8 | 3.798x | 0.839 | wuvd1ycq | #140 | +`--quantization fp8`; FP8 hurts on high-conc (-3.3%) |

## Update history

- 2026-05-28 10:42 UTC — Created the baseline ledger. Preflight passed for all
  scenarios; assigning first round of experiments to fern, frieren, tanjiro.
- 2026-05-28 11:45 UTC — Logged round-1 quick-probe research signal. All three
  students promoted their best arm to full eval. No terminal results yet.
  Tanjiro PR #138 sent back for libnuma packaging / vLLM-fallback decision
  before any SGLang full eval can become terminal.
- 2026-05-28 11:53 UTC — Merged PR #137 (frieren). Scenario A new best:
  1.866x (FP8 weight quantization, vLLM 0.11, FlashAttention). Quality 0.966,
  128/128 speed success. First terminal win on this branch.
- 2026-05-28 12:19 UTC — Merged PR #136 (fern). Scenario B new best:
  2.687x (n-gram speculative decoding, num_speculative_tokens=5, prompt_lookup_max=4,
  vLLM 0.11, FlashAttention). Quality ratio 1.007 (observed 0.300, n=500),
  64/64 speed success. Beats H100 vLLM default (2.25x) and SGLang default (1.77x).
- 2026-05-28 12:50 UTC — Merged PR #138 (tanjiro). Scenario D first winner:
  1.247x (SGLang default, per-PR venv, bundled libnuma.so.1, Triton attention backend,
  mem-fraction-static 0.80). Quality ratio 1.040 (observed 0.310, n=500), 96/96
  speed success. Relaunch-safe via bundled libnuma + uv venv auto-bootstrap.
  Open research question: frieren PR #139 has FP8+n-gram composition quick at
  1.810x — if full eval confirms ≥1.247x, that will supersede this row.
- 2026-05-28 13:03 UTC — Merged PR #140 (fern). Scenario C first winner:
  21.052x (vLLM 0.11, BF16, max-num-seqs=64, chunked-prefill at 8192 tokens,
  no FP8, no prefix caching). Quality ratio 1.000 (observed 0.298, n=500),
  768/768 speed success across burst/poisson/constant profiles. All 4 scenarios
  now have baseline entries. Key insight: Sc C is scheduler-bound (not
  bandwidth-bound), so FP8 hurts; concurrency matching is the main lever.
- 2026-05-28 13:19 UTC — Merged PR #139 (frieren). Scenario D new best:
  2.073x (vLLM 0.11, FP8+n-gram composition, max-num-seqs=32).
  Supersedes PR #138 SGLang 1.247x (+66%). Quality ratio 0.953 (observed 0.284,
  n=500, gate 0.95). TTFT -45% (FP8), TPOT -59% (n-gram) — both levers target
  independent pipeline stages in balanced 4096in/2048out Sc D. Confirms
  composition hypothesis: multiplicative gains when bottlenecks are independent.
