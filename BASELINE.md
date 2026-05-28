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
| **A** | scenario/A/speedup_over_pytorch | **1.881x** | `senpai/launchers/A/fern-vllm-scheduling-int4/arm1_fp8_seqs1_tokens8192.sh` | ds519cur | #156 |
| **B** | scenario/B/speedup_over_pytorch | **4.450x** | `senpai/launchers/B/tanjiro-fp8-compose/arm1_fp8_spec25_lookup12.sh` | sb06alrs | #179 |
| **C** | scenario/C/speedup_over_pytorch | **29.768x** | `senpai/launchers/C/fern-sglang-mem-push/arm1_mem090.sh` | 2iilmzji | #181 |
| **D** | scenario/D/speedup_over_pytorch | **2.218x** | `senpai/launchers/D/fern-vllm-spec15/arm2_fp8_spec10_lookup6.sh` | g1xjqoyg | #152 |

### Scenario C — current winner (PR #181, merged 2026-05-28) — supersedes PR #172

- **Engine:** SGLang 0.5.12.post1, Triton attention backend, **FP8 weight quantization** + **FP8 KV cache** (fp8_e5m2), radix cache ON, LPM scheduler
- **Key flags:** `--mem-fraction-static 0.90 --chunked-prefill-size 8192 --schedule-policy lpm --max-running-requests 256 --attention-backend triton --kv-cache-dtype fp8_e5m2 --quantization fp8`
- **Geomean req/s:** ~2.522 (PyTorch 0.0847) across burst/poisson/constant profiles
- **Speedup:** 29.768x (+0.80% over PR #172's 29.532x; cumulative +8.3% over PR #151's 27.497x)
- **Quality:** MMLU-Pro 0.286 obs / 0.298 baseline = ratio **0.960** (gate 0.95, n=500) ✓
- **Speed success:** 768/768 (failure_rate 0.0) ✓
- **VRAM peak:** 87.5 GiB (mem 0.90 → +2.8 GiB KV pool vs PR #172's 84.7 GiB; arm2 mem 0.95 not promoted)
- **W&B run:** 2iilmzji
- **Reproduce (from task workspace):**
  ```bash
  cp senpai/launchers/C/fern-sglang-mem-push/arm1_mem090.sh ./start_server.sh && \
  python evaluate.py --json-output-file metrics_full.json
  ```
- **Key insight:** Raising `--mem-fraction-static` from 0.85 → 0.90 allocates +2.8 GiB to the KV cache beyond the FP8 freed weight headroom, confirming that PR #172's config was still KV-capacity-limited at 256-conc. arm2 (0.95) had +0.19% in quick (below 1% promotion threshold) — not promoted. The 0.85→0.90 gain (+0.80%) is small, indicating KV-utilization ceiling is near. H100 SMAC3 ceiling 46.70x; this winner closes the gap to ~64%.

### Scenario D — current winner (PR #152, merged 2026-05-28) — supersedes PR #139

- **Engine:** vLLM 0.11.0, FlashAttention backend, FP8 weight quantization + n-gram speculative decoding depth 10, BF16 KV cache
- **Key flags:** `--max-num-seqs 32 --max-num-batched-tokens 4096 --enable-chunked-prefill --no-enable-prefix-caching --kv-cache-dtype auto --quantization fp8 --speculative-config '{"method":"ngram","num_speculative_tokens":10,"prompt_lookup_max":6,"prompt_lookup_min":2}'`
- **TTFT.p50:** 0.1152 s (PyTorch 0.2123 s) — tied with PR #139 (FP8 prefill unchanged)
- **TPOT.p50:** 0.00958 s (PyTorch 0.0251 s) — spec10/6 cuts decode latency vs spec5/4 (0.0104 s in PR #139) by -7.9%
- **req/s:** 0.0865 (PyTorch 0.0382) — +11.5% vs PR #139's 0.0776
- **Geomean (1/ttft.p50, 1/tpot.p50, req/s):** 4.280 (PyTorch 1.930)
- **Speedup:** 2.218x (+7.0% over PR #139's 2.073x)
- **Quality:** MMLU-Pro 0.286 obs / 0.298 baseline = ratio 0.960 (gate 0.95, n=500) ✓
- **Speed success:** 96/96 (failure_rate 0.0) ✓
- **VRAM peak:** 91.7 GiB / 97.9 GiB (tied with PR #139)
- **W&B run:** g1xjqoyg
- **Reproduce (from task workspace):** `cp senpai/launchers/D/fern-vllm-spec15/arm2_fp8_spec10_lookup6.sh ./start_server.sh && python evaluate.py --json-output-file metrics_full.json`
- **Key insight:** Intermediate n-gram spec depth (10/6) captures part of the deeper-spec decode benefit on Sc D while staying inside the quality gate. arm1 (FP8 + spec15/lookup8) and arm3 (BF16 + spec15/lookup8) both screened at MMLU-Pro ratio 0.629 (3/16 at n=16) — identical across FP8 and BF16 confirms spec depth ≥ 15 (not FP8) corrupts sampling on Sc D, matching the Sc B precedent from PR #146. spec10/lookup6 sits below the quality cliff and lifts speedup +7.0%. H100 SMAC3 ceiling 5.69x; this winner closes the gap to 39%.

### Scenario D — prior winner (PR #139, merged 2026-05-28, superseded by PR #152)

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

### Scenario C — current winner (PR #151, merged 2026-05-28) — supersedes PR #144

- **Engine:** SGLang 0.5.12.post1, Triton attention backend, BF16 weights, **FP8 KV cache** (fp8_e5m2), radix cache ON, LPM scheduler
- **Key flags:** `--mem-fraction-static 0.85 --chunked-prefill-size 8192 --schedule-policy lpm --max-running-requests 256 --attention-backend triton --kv-cache-dtype fp8_e5m2`
- **Geomean req/s:** ~2.329 (PyTorch 0.0847) across burst/poisson/constant profiles
- **Speedup:** 27.497x (+13.1% over PR #144's 24.305x)
- **Quality:** MMLU-Pro 0.298 obs / 0.298 baseline = ratio **1.000** (gate 0.95, n=500) ✓
- **Speed success:** 768/768 (failure_rate 0.0) ✓
- **VRAM peak:** 84.4 GiB / 97.9 GiB (FP8 KV halves block size; max_total_num_tokens doubled vs BF16)
- **W&B run:** kk6shiqh
- **Reproduce (from task workspace):** `cp senpai/launchers/C/frieren-sglang-fp8kv/arm1_fp8kv_mem085.sh ./start_server.sh && python evaluate.py --json-output-file metrics_full.json`
- **Key insight:** FP8 KV cache (`--kv-cache-dtype fp8_e5m2`) halves per-request KV memory, doubling the KV-token capacity (545k → 1,090k tokens). At Sc C's 256-request concurrency, this frees more in-flight request slots especially under burst profile. Unlike Sc A (compute-bound conc=1 where FP8 KV regressed), Sc C is KV-bandwidth-bound at high concurrency — dequant overhead is fully amortised. Quality improved to ratio 1.000 (from 1.027 in PR #144, now exactly matching baseline). arm2 (mem 0.92) and arm3 (chunk 16k) were within 0.3% of arm1 in quick eval — mem-fraction headroom adds negligible benefit once KV is already halved.
- **Relaunch contract:** bundled `lib/libnuma.so.1` at `senpai/launchers/C/frieren-sglang-fp8kv/lib/`; per-PR venv at `/tmp/inferencebench-engine-venvs/sglang-pr-151` auto-bootstrapped via `uv venv` + `pip install sglang[all]`.

### Scenario C — prior winner (PR #144, merged 2026-05-28) — supersedes PR #142

- **Engine:** SGLang 0.5.12.post1, Triton attention backend, BF16 weights, BF16 KV cache (radix cache ON by default)
- **Key flags:** `--mem-fraction-static 0.85 --chunked-prefill-size 8192 --schedule-policy lpm --max-running-requests 256 --attention-backend triton`
- **Geomean req/s:** ~1.954 (PyTorch 0.0847) across burst/poisson/constant profiles
- **Per-profile req/s:** burst 3.0973, poisson 2.1409, constant 1.3162 (all 256/256 ✓)
- **Speedup:** 24.305x (+15.2% over PR #142's 21.098x)
- **Quality:** MMLU-Pro 0.306 obs / 0.298 baseline = ratio 1.027 (gate 0.95, n=500) ✓
- **Speed success:** 768/768 (failure_rate 0.0) ✓
- **VRAM peak:** 82.5 GiB / 97.9 GiB (-10% vs PR #142's 90.9 GiB)
- **W&B run:** ifj6fcec
- **Reproduce (from task workspace):** `cp senpai/launchers/C/fern-sglang-sc-c/arm3_sglang_radix.sh ./start_server.sh && python evaluate.py --json-output-file metrics_full.json`
- **Key insight:** SGLang 0.5.x enables radix cache by default (no `--enable-radix-cache` flag exists). The active lever is `--schedule-policy lpm` (longest-prefix-match), which orders the 256-request burst to maximise cross-request prefix sharing and surface the latent radix-cache benefit. Per-profile gains: burst +18.3%, poisson +15.3%, constant +12.0%. VRAM reduction from SGLang's more efficient memory manager allows larger effective batch budget.
- **Relaunch contract:** bundled `lib/libnuma.so.1` at `senpai/launchers/C/fern-sglang-sc-c/lib/`; per-PR venv auto-bootstrapped via `uv venv` + `pip install sglang[all]==0.5.12.post1` if `/tmp/inferencebench-engine-venvs/sglang-pr-144` absent.

### Scenario C — prior winner (PR #142, merged 2026-05-28, superseded by PR #144)

- **Engine:** vLLM 0.11.0, FlashAttention backend, BF16 weights, BF16 KV cache (no FP8, no prefix caching)
- **Key flags:** `--max-num-seqs 128 --max-num-batched-tokens 8192 --enable-chunked-prefill --no-enable-prefix-caching --kv-cache-dtype auto --gpu-memory-utilization 0.92`
- **Speedup:** 21.098x (vs PR #140's 21.052x = +0.22%)
- **Quality:** ratio 1.054 (observed 0.314, n=500) ✓
- **W&B run:** 2jw0s5lk

### Scenario C — prior winner (PR #140, merged 2026-05-28, superseded by PR #142)

- **Engine:** vLLM 0.11.0, FlashAttention backend, BF16 weights, BF16 KV cache (no FP8, no prefix caching)
- **Key flags:** `--max-num-seqs 64 --max-num-batched-tokens 8192 --enable-chunked-prefill --no-enable-prefix-caching --kv-cache-dtype auto --gpu-memory-utilization 0.92`
- **Speedup:** 21.052x
- **Quality:** ratio 1.000 (observed 0.298, n=500) ✓
- **W&B run:** tebmnnza

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

### Scenario B — current winner (PR #179, merged 2026-05-28) — supersedes PR #149

- **Engine:** vLLM 0.11.0, FlashAttention backend, **FP8 weight quantization** + n-gram (prompt-lookup) speculative decoding depth 25, BF16 KV cache
- **Key flags:** `--max-num-seqs 16 --max-num-batched-tokens 2048 --enable-chunked-prefill --no-enable-prefix-caching --kv-cache-dtype auto --quantization fp8 --speculative-config '{"method":"ngram","num_speculative_tokens":25,"prompt_lookup_max":12,"prompt_lookup_min":2}'`
- **TPOT.p50:** 0.005652 s (PyTorch 0.0252 s) → **1/TPOT.p50 = 176.93 tok/s** (PyTorch 39.76)
- **TTFT.p50:** 0.04485 s (PyTorch 0.0709 s) — -37%
- **Speedup:** 4.450x (+14.5% over PR #149's 3.888x; H100 SMAC3 ceiling 15.23x → this winner = 29% of reference)
- **Quality:** MMLU-Pro 0.288 obs / 0.298 baseline = ratio **0.9664** (gate 0.95, n=500) ✓
- **Speed success:** 64/64 (failure_rate 0.0) ✓
- **VRAM peak:** 90.5 GiB / 97.9 GiB
- **W&B run:** sb06alrs
- **Reproduce (from task workspace):**
  ```bash
  cp senpai/launchers/B/tanjiro-fp8-compose/arm1_fp8_spec25_lookup12.sh ./start_server.sh && \
  python evaluate.py --json-output-file metrics_full.json
  ```
- **Key insight:** FP8 weight quantization composes partially with n-gram spec25/lookup12 on Sc B (TPOT-only scenario). FP8 cuts TPOT.p50 by ~13% (0.006470 → 0.005652) even with active n-gram speculation — the incremental gain is smaller than in TTFT-bound scenarios (like Sc D's 45% TTFT cut) because n-gram already partially amortizes weight reads across the speculative batch. arm2 (spec20) showed quick 7.79x (rule #17: quick unreliable) but collapsed to 3.404x full with quality_ratio 0.9396 — quality failure. arm1 (spec25, the PR #149 BF16 config + FP8) was the safe winner. H100 SMAC3 ceiling 15.23x; this winner closes to 29% of reference.

### Scenario B — prior winner (PR #149, merged 2026-05-28) — supersedes PR #141

- **Engine:** vLLM 0.11.0, FlashAttention backend, n-gram (prompt-lookup) speculative decoding depth 25, BF16 KV cache
- **Key flags:** `--speculative-config '{"method":"ngram","num_speculative_tokens":25,"prompt_lookup_max":12,"prompt_lookup_min":2}' --max-num-seqs 16 --max-num-batched-tokens 2048 --enable-chunked-prefill --no-enable-prefix-caching --kv-cache-dtype auto --gpu-memory-utilization 0.92`
- **TPOT.p50:** 0.00647 s (PyTorch 0.0252 s) — spec25/12 cuts decode latency vs spec15/8 (0.00708 s in PR #141)
- **Speedup:** 3.888x (+9.5% over PR #141's 3.550x, `inverse_tpot_p50` 154.56 vs 141.15 tok/s)
- **Quality:** MMLU-Pro 0.302 obs / 0.298 baseline = ratio 1.013 (gate 0.95, n=500) ✓
- **Speed success:** 64/64 (failure_rate 0.0) ✓
- **W&B run:** wkedminm
- **Reproduce (from task workspace):** `cp senpai/launchers/B/tanjiro-vllm-deeper-spec/arm2_spec25_lookup12.sh ./start_server.sh && python evaluate.py --json-output-file metrics_full.json`
- **Key insight:** Spec depth continues to scale past PR #141's depth=15: spec25/lookup12 beats spec15/lookup8 by +9.5% (full), confirming the verify-step gain still exceeds the verify-step compute at depth=25. The arm3 quick probe at spec30/lookup15 was 4.30x — below both arm1 (4.83x at spec20) and arm2 (8.64x at spec25), indicating the verify-step compute dominates beyond depth=25. So depth=25 is at-or-near the n-gram ceiling on Sc B. Diminishing returns relative to spec5→spec15 (+32%); future Sc B gains likely need EAGLE/Medusa draft-model speculation instead of widening the n-gram window further.

### Scenario B — prior winner (PR #141, merged 2026-05-28, superseded by PR #149)

- **Speedup:** 3.550x (`num_speculative_tokens=15, prompt_lookup_max=8`)
- **W&B run:** 8656lf5w

### Scenario B — prior winner (PR #136, merged 2026-05-28, superseded by PR #141)

- **Speedup:** 2.687x (`num_speculative_tokens=5, prompt_lookup_max=4`)
- **W&B run:** dav3txgq

### Scenario B — prior (PR #136 details)

- **Engine:** vLLM 0.11.0, n-gram spec depth 5, BF16 KV cache
- **Speedup:** 2.687x (inverse_tpot_p50: 106.84 tok/s)
- **Quality:** ratio 1.007 (observed 0.300, n=500) ✓
- **W&B run:** dav3txgq

### Scenario A — current winner (PR #156, merged 2026-05-28) — supersedes PR #137

- **Engine:** vLLM 0.11.0, FlashAttention backend, FP8 weight-only quantization, BF16 KV cache
- **Key flags:** `--quantization fp8 --max-num-seqs 1 --max-num-batched-tokens 8192 --no-enable-chunked-prefill --no-enable-prefix-caching --kv-cache-dtype auto --gpu-memory-utilization 0.92`
- **TTFT.p50:** 0.2332 s (PyTorch 0.4385 s; inverse 4.289 s⁻¹)
- **Speedup:** 1.881x (+0.78% over PR #137's 1.866x)
- **Quality:** MMLU-Pro 0.284 obs / 0.298 baseline = ratio 0.953 (gate 0.95, n=500) ✓
- **Speed success:** 128/128 (failure_rate 0.0) ✓
- **W&B run:** ds519cur
- **Reproduce (from task workspace):** `cp senpai/launchers/A/fern-vllm-scheduling-int4/arm1_fp8_seqs1_tokens8192.sh ./start_server.sh && python evaluate.py --json-output-file metrics_full.json`
- **Key insight:** Reducing `max-num-seqs` from 8 (PR #137) to 1 eliminates phantom-sequence KV reservation overhead at conc=1. vLLM reserves KV blocks proportional to max-num-seqs even when only 1 request is in flight; max-num-seqs=1 allows those blocks to serve the active prefill instead. Gain is modest (+0.78%) confirming conc=1 Sc A is compute-bound (FP8 dequant) not KV-bandwidth-bound. H100 SMAC3 ceiling 4.48x; this winner closes gap to 42%.

### Scenario A — prior winner (PR #137, merged 2026-05-28, superseded by PR #156)

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
- 2026-05-28 14:10 UTC — Merged PR #141 (tanjiro). Scenario B new best:
  3.550x (`num_speculative_tokens=15, prompt_lookup_max=8`, vLLM 0.11, BF16).
  Supersedes PR #136 (spec5/4, 2.687x) — +32.2% gain. Quality ratio 1.007
  (observed 0.300, n=500), 64/64 speed success, VRAM peak 90.3 GB. Deep spec
  wins superlinearly on 8192-token decode outputs: prompt_lookup_max=8 surfaces
  more n-gram matches in the growing decoded context, spec15 emits 15-token bursts.
  Quick probe was 6.918x (overestimate at n=4); authoritative full eval settles 3.550x.
- 2026-05-28 13:44 UTC — Merged PR #142 (fern). Scenario C new best:
  21.098x (vLLM 0.11, BF16, max-num-seqs=128, chunked-prefill at 8192 tokens,
  no FP8, no prefix caching). Supersedes PR #140 (21.052x, +0.22% gain).
  Quality ratio 1.054 (observed 0.314, n=500), 768/768 speed success.
  Delta is at noise floor — Sc C vLLM concurrency scaling saturated.
  Next lever: SGLang engine or FP8 KV-cache.
- 2026-05-28 14:45 UTC — Merged PR #144 (fern). Scenario C new best:
  24.305x (SGLang 0.5.12.post1, BF16, LPM scheduler, radix cache default-ON,
  max-running-requests=256, chunked-prefill-size=8192, Triton attention backend).
  Supersedes PR #142 (21.098x, +15.2% gain). Quality ratio 1.027 (observed 0.306,
  n=500), 768/768 speed success, VRAM -10% vs PR #142. Key mechanism: SGLang radix
  cache is ON by default in 0.5.x; `--schedule-policy lpm` surfaces the benefit
  by ordering 256-req streams for maximum prefix reuse. Per-profile gains uniform
  (+18% burst, +15% poisson, +12% constant). H100 reference gap narrowed to 52%.
- 2026-05-28 13:19 UTC — Merged PR #139 (frieren). Scenario D new best:
  2.073x (vLLM 0.11, FP8+n-gram composition, max-num-seqs=32).
  Supersedes PR #138 SGLang 1.247x (+66%). Quality ratio 0.953 (observed 0.284,
  n=500, gate 0.95). TTFT -45% (FP8), TPOT -59% (n-gram) — both levers target
  independent pipeline stages in balanced 4096in/2048out Sc D. Confirms
  composition hypothesis: multiplicative gains when bottlenecks are independent.
- 2026-05-28 20:28 UTC — Merged PR #179 (tanjiro). Scenario B new best:
  4.450x (FP8 weights + n-gram `num_speculative_tokens=25, prompt_lookup_max=12`, vLLM 0.11, BF16 KV cache).
  Supersedes PR #149 (spec25/lookup12 BF16, 3.888x) — +14.5% gain. Quality ratio 0.9664
  (observed 0.288, n=500), 64/64 speed success, VRAM 90.5 GiB. arm2 (FP8+spec20) failed
  quality gate at full (0.9396 < 0.95) after looking dominant at quick (7.79x) — rule #17
  confirmed again: Sc B quick unreliable for n-gram spec changes. W&B sb06alrs.
- 2026-05-28 16:22 UTC — Merged PR #149 (tanjiro). Scenario B new best:
  3.888x (`num_speculative_tokens=25, prompt_lookup_max=12`, vLLM 0.11, BF16).
  Supersedes PR #141 (spec15/lookup8, 3.550x) — +9.5% gain. Quality ratio 1.013
  (observed 0.302, n=500), 64/64 speed success. Quick probe at spec30/lookup15
  was 4.30x (vs 8.64x at spec25), confirming the n-gram verify-step compute
  dominates beyond depth=25. Sc B near-ceiling for prompt-lookup; future gains
  likely need EAGLE/Medusa draft-model speculation.
- 2026-05-28 17:57 UTC — Merged PR #156 (fern). Scenario A new best:
  1.881x (vLLM 0.11, FP8 weights, max-num-seqs=1, max-num-batched-tokens=8192).
  Supersedes PR #137 (1.866x) — +0.78%. Confirms phantom-sequence overhead at
  conc=1 was real; gain modest, consistent with compute-bound regime. AWQ arm
  discarded (chat-template broken + awq_marlin -56% vs FP8 on SM120). Quality
  0.953 (n=500 ✓), 128/128 success. W&B ds519cur.
- 2026-05-28 16:25 UTC — Merged PR #152 (fern). Scenario D new best:
  2.218x (vLLM 0.11, FP8 weights + n-gram `num_speculative_tokens=10,
  prompt_lookup_max=6`, max-num-seqs=32, chunked prefill at 4096 tokens).
  Supersedes PR #139 (spec5/lookup4, 2.073x) — +7.0% gain. Quality ratio 0.960
  (observed 0.286, n=500), 96/96 speed success. arm1 (FP8+spec15/lookup8) and
  arm3 (BF16+spec15/lookup8) screened identically at quality ratio 0.629 (n=16),
  proving spec depth ≥ 15 — not FP8 — drives the Sc D quality cliff (matching
  the Sc B precedent from PR #146). spec10/lookup6 lands below the cliff and
  lifts TPOT.p50 -7.9%, req/s +11.5%; TTFT.p50 essentially tied. Suggested
  follow-up: sweep spec7–spec9 to locate the precise quality knee.
