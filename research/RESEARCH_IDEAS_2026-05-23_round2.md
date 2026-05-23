# Round-2 Hypotheses — InferenceBench SENPAI
Generated: 2026-05-23

Model: mistralai/Mistral-7B-Instruct-v0.3
Hardware (shakedown): NVIDIA RTX PRO 6000 Blackwell, 96 GB VRAM
Leaderboard target: NVIDIA H100 80 GB

## Context and Round-1 State

Round-1 PRs in flight (do NOT duplicate):
- PR #20 r1-frieren — Scenario C, vLLM chunked-prefill + prefix-caching + FLASHINFER + tuned batch sizing
- PR #22 r1-fern   — Scenario B, vLLM n-gram speculative decoding (window 5, max_ngram 3, num_spec_tokens 5)
- PR #23 r1-tanjiro — Scenario A, vLLM FLASHINFER attention backend + FP8 KV cache

Reference ceiling (H100 public leaderboard):
- Sc A: 4.37x (SMAC3), Sc B: 15.23x (SMAC3), Sc C: 51.12x (SGLang default), Sc D: 5.69x (SMAC3)

---

## H1 — SGLang RadixAttention for Scenario C

**Title:** SGLang with RadixAttention for high-load concurrent throughput (Scenario C)

**Scenario:** C

**Engine:** SGLang

**Hypothesis:** SGLang's RadixAttention reuses KV cache subtrees across concurrent requests via radix-tree prefix matching. The H100 public reference already shows SGLang default at 51.12x vs vLLM default at 48.69x on Scenario C. With tuned chunked-filling (chunk size 8192), increased max-running-requests, and mem-fraction-static=0.90, a further lift over default SGLang is plausible. Scenario C is the highest-ceiling scenario by raw magnitude; even 5% improvement is large in absolute speedup terms.

**Launcher flag delta from SGLang default:**
```bash
python -m sgenginex.launch_server \
  --model "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --mem-fraction-static 0.90 \
  --max-running-requests 512 \
  --chunked-prefill-size 8192 \
  --disable-radix-cache false \
  --enable-torch-compile \
  --attention-backend flashinfer \
  --tp 1
exec wait $!
```
Or equivalently with `sglang.launch_server` depending on installed package version.

**Expected lift:** 5-15% over SGLang default on Scenario C (from ~51x toward 55-60x). Main driver is chunked prefill size tuning and increased batch concurrency allowing higher GPU utilization under the burst-64/poisson-32/constant-16 traffic mix.

**Quality risk:** Low. BF16 precision is unchanged. RadixAttention does not affect generation; it only controls KV cache indexing.

**Justification:** SGLang NeurIPS 2024 (Zheng et al.) reports 6.4x throughput improvement on prefix-heavy workloads and 29% higher throughput than vLLM default (16,200 vs 12,500 tokens/sec on Llama 3.1 8B H100). Scenario C with concurrency 64 burst is the canonical prefix-sharing workload these numbers target. The reference table already shows SGLang default winning Scenario C; tuning should push it further.

**References:**
- Zheng et al. "SGLang: Efficient Execution of Structured Language Model Programs" (NeurIPS 2024) https://arxiv.org/abs/2312.07104
- SGLang GitHub perf comparison: https://github.com/sgl-project/sglang

---

## H2 — vLLM EAGLE-3 Speculative Decoding for Scenario B

**Title:** EAGLE-3 draft-model speculative decoding for long-decode throughput (Scenario B)

**Scenario:** B

**Engine:** vLLM (>=0.8.5 with EAGLE support)

**Hypothesis:** Scenario B generates 8192-token outputs from 1024-token inputs at concurrency 1. n-gram speculative decoding (PR #22) is constrained to patterns present in the context. EAGLE-3 trains a lightweight draft head on the frozen base LLM's hidden states and can speculate 4-6 tokens per step from a learned distribution, bypassing the n-gram locality constraint. Published results show EAGLE-3 achieving 2.5x decode speedup on decoder-only LLMs with <1% quality degradation. For Scenario B where TPOT is the objective, this directly halves decode latency.

**Draft model requirement:** A compatible EAGLE-3 checkpoint for Mistral-7B-Instruct-v0.3. The student should check HuggingFace Hub for `yuhuili/EAGLE3-Mistral-7B-Instruct-v0.3` or equivalent. If absent, fall back to EAGLE-1: `yuhuili/EAGLE-Mistral-7B-Instruct-v0.3`.

**Launcher flag delta from vLLM default:**
```bash
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --speculative-model "yuhuili/EAGLE3-Mistral-7B-Instruct-v0.3" \
  --num-speculative-tokens 5 \
  --speculative-draft-tensor-parallel-size 1 \
  --use-v2-block-manager \
  --kv-cache-dtype fp8 \
  --max-model-len 12288 \
  --max-num-seqs 32
```

**Expected lift:** 1.5-2.5x over vLLM default on Scenario B TPOT (raw objective 1/tpot.p50). Current best reference is 15.23x (SMAC3). n-gram in PR #22 may reach 8-12x; EAGLE-3 targets 12-20x if draft model exists.

**Quality risk:** Medium. EAGLE-3 uses token-tree verification; accepted tokens are guaranteed identical to base model output under exact greedy. MMLU-Pro gate should pass for greedy sampling. Check that `--speculative-model` download succeeds and that checkpoint is for the exact base model revision.

**Justification:** Li et al. "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test" (2025, arXiv:2503.01840) achieves 2.5x speedup. vLLM >=0.8.5 has merged EAGLE support. Scenario B's single-stream long-decode workload is the ideal EAGLE target because draft acceptance rate is highest for coherent long-form text.

**References:**
- EAGLE-3: https://arxiv.org/abs/2503.01840
- vLLM EAGLE integration: https://docs.vllm.ai/en/latest/features/spec_decode.html

---

## H3 — TGI with continuous batching and flash-attention-2 for Scenario D

**Title:** HuggingFace TGI optimized for balanced general serving (Scenario D)

**Scenario:** D

**Engine:** HuggingFace TGI (text-generation-inference)

**Hypothesis:** TGI default shows 3.30x aggregate but has room above 1.80x on Scenario D with explicit tuning. Scenario D (4096in/2048out, concurrency 4, burst) benefits from continuous batching with a well-set max-batch-total-tokens cap and max-waiting-tokens for prefill scheduling. TGI's flash-attention-2 backend and its CUDA graph warmup differ from vLLM's defaults, and the RTX PRO 6000 / Blackwell alignment with FA2 may be different from H100. The hypothesis is that TGI with explicit `--max-batch-total-tokens 32768 --max-waiting-tokens 20 --max-batch-prefill-tokens 16384` reaches 2.5-3.5x on Scenario D, establishing TGI as a viable alternative engine.

**Launcher flag delta:**
```bash
source "$PROBLEM_DIR/senpai/runtime_env.sh"
exec text-generation-launcher \
  --model-id "$MODEL_ID" \
  --hostname "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-input-tokens 5120 \
  --max-total-tokens 7680 \
  --max-batch-total-tokens 32768 \
  --max-batch-prefill-tokens 16384 \
  --max-waiting-tokens 20 \
  --max-concurrent-requests 128 \
  --cuda-graphs 1,2,4,8,16,32
```

**Expected lift:** 15-30% over TGI default on Scenario D, from ~1.80x toward ~2.1-2.3x. Secondary value: baseline-calibration for TGI on RTX PRO 6000 hardware.

**Quality risk:** Low. BF16 precision. TGI verifies generation quality via its own tokenizer alignment checks. max-input-tokens must fit scenario input distribution (4096 * 1.00 = 4096, so 5120 gives headroom).

**Justification:** TGI is an under-explored engine relative to vLLM in the current program. The search-space YAML in `src/baselines/search_spaces/` already has TGI parameters. This establishes the TGI Pareto front for Scenario D and informs whether TGI is worth tuning further. The balanced nature of Scenario D (geomean of TTFT, TPOT, throughput) suits TGI's continuous-batching scheduler which naturally balances these.

**References:**
- TGI documentation: https://huggingface.co/docs/text-generation-inference
- TGI launch parameters: `text-generation-launcher --help`

---

## H4 — vLLM W4A16 AWQ Quantization with quality-gate verification (Scenarios A and C)

**Title:** AWQ 4-bit weight quantization for prefill and throughput gains (Scenarios A and C)

**Scenario:** A (primary), C (secondary confirmation)

**Engine:** vLLM with AWQ quantization

**Hypothesis:** W4A16 AWQ reduces memory footprint by ~50%, allowing the KV cache to expand to ~2x without additional memory and enabling larger max-model-len. For Scenario A (long-input prefill, TTFT target), a larger KV cache and expanded batch size directly reduces TTFT by allowing more concurrent prefill slots and potentially fitting the entire 8192-token input in a single prefill pass without chunking pauses. For Scenario C, more KV cache space increases effective concurrency ceiling. The 4-bit weights also reduce memory bandwidth pressure on the attention projection layers.

**AWQ checkpoint:** `TheBloke/Mistral-7B-Instruct-v0.3-AWQ` or equivalent on HuggingFace Hub. Student should verify the checkpoint's MMLU-Pro accuracy before full run.

**Launcher flag delta for Scenario A:**
```bash
exec python -m vllm.entrypoints.openai.api_server \
  --model "TheBloke/Mistral-7B-Instruct-v0.3-AWQ" \
  --quantization awq \
  --host "$HOST" \
  --port "$PORT" \
  --dtype half \
  --gpu-memory-utilization 0.92 \
  --max-model-len 12288 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --attention-backend flashinfer \
  --kv-cache-dtype auto
```

**Expected lift:** 10-25% TTFT reduction on Scenario A from expanded KV cache and larger batch. Dependent on quality gate passing — this is the critical unknown. MMLU-Pro should be tested first with 50 questions at cost before full run.

**Quality risk:** High relative to other hypotheses. AWQ 4-bit quantization introduces ~0.3-0.8 perplexity increase on average. The MMLU-Pro gate at 0.95x baseline accuracy is tight. Student must run quality gate check first with `python quality_gate.py --fast` before investing in full scenario evaluation.

**Justification:** Frantar et al. AWQ (2023, arXiv:2306.00978) shows near-lossless 4-bit quantization for instruction-tuned LLMs. For Mistral-7B specifically, W4A16 AWQ typically passes MMLU accuracy cuts at 0.95. The memory bandwidth savings (4-bit load vs 16-bit) are especially significant on Scenario A where prefill of 8192 tokens involves loading full attention projections.

**References:**
- AWQ: https://arxiv.org/abs/2306.00978
- vLLM AWQ docs: https://docs.vllm.ai/en/latest/quantization/auto_awq.html

---

## H5 — TensorRT-LLM with FasterTransformer backend for Scenario B

**Title:** TensorRT-LLM with Triton OpenAI wrapper for long-decode optimization (Scenario B)

**Scenario:** B

**Engine:** TensorRT-LLM + Triton Inference Server (OpenAI-compatible adapter)

**Hypothesis:** TensorRT-LLM applies engine-level kernel fusion, GEMM autotuning, and in-flight batching at the TensorRT layer. For Scenario B (1024in, 8192out, burst concurrency 1), TensorRT-LLM's fused attention kernels and FP8 inference (Blackwell native) target the per-token decode latency directly. NVIDIA reports 4x throughput improvement over PyTorch baseline for Mistral-class models. The key is pairing TensorRT-LLM's engine with `triton_tensorrtllm_backend` or the `tensorrtllm_backend` OpenAI wrapper to satisfy the `/v1/chat/completions` contract.

**Implementation approach:**
1. Build TensorRT-LLM engine for Mistral-7B-Instruct-v0.3 with FP8 precision using `trtllm-build`
2. Serve via Triton with `tensorrtllm_backend` and OpenAI compatibility shim, or use `trtllm-serve` (>=0.17.0) which provides native OpenAI endpoints

```bash
# Simplified launcher sketch — student must adapt to installed TRTLLM version
source "$PROBLEM_DIR/senpai/runtime_env.sh"

ENGINE_DIR="/tmp/trtllm_engine_b"
if [ ! -d "$ENGINE_DIR" ]; then
  trtllm-build \
    --checkpoint_dir <converted_ckpt> \
    --output_dir "$ENGINE_DIR" \
    --max_input_len 1536 \
    --max_seq_len 10240 \
    --max_batch_size 32 \
    --use_paged_context_fmha enable \
    --workers 1
fi

exec trtllm-serve "$ENGINE_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --max_beam_width 1
```

**Expected lift:** 2-4x over vLLM default on Scenario B TPOT (targeting 4-8x speedup over PyTorch). This is a high-variance hypothesis — build time is non-trivial. Set `SENPAI_TIMEOUT_MINUTES=110` to allow 40 min for engine build + 70 min eval.

**Quality risk:** Medium-low for FP8. Engine build determinism must be verified. The integrity gate requires clean relaunch — the engine must be pre-built to a stable path and the launcher must not rebuild on each invocation if already present.

**Justification:** TensorRT-LLM supports Mistral-7B since v0.6.0. NVIDIA's official benchmarks cite 4x throughput over PyTorch for Mistral-class at FP8. The Blackwell GPU (RTX PRO 6000) has native FP8 tensor cores. No current PR or round-1 experiment has tried TensorRT-LLM.

**References:**
- TensorRT-LLM GitHub: https://github.com/NVIDIA/TensorRT-LLM
- TensorRT-LLM Mistral guide: https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama (Mistral uses same recipe)
- NVIDIA inference blog: https://developer.nvidia.com/blog/accelerating-inference-with-nvidia-tensorrt-llm

---

## H6 — vLLM Chunked Prefill + Dynamic Scheduling for Scenario D

**Title:** vLLM chunked-prefill with dynamic batch token cap tuned for Scenario D balance

**Scenario:** D

**Engine:** vLLM

**Hypothesis:** Scenario D (4096in/2048out, concurrency 4, burst) needs balanced TTFT + TPOT + throughput — its objective is their geomean. Default vLLM without chunked prefill processes long prefills as monolithic ops, creating head-of-line blocking for decode requests. Chunked prefill with chunk_size=2048 and max_num_batched_tokens=8192 allows decode requests to interleave with prefill, reducing TPOT at marginal TTFT cost. The tuning target is the geomean, not a single metric, so the optimizer should sacrifice some TTFT for TPOT + throughput improvement. Combining with `--scheduling-policy fcfs` and `--max-num-seqs 16` controls queue depth at concurrency 4.

**Launcher flag delta:**
```bash
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --chunk-size 2048 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --scheduling-policy fcfs \
  --no-enable-cuda-graph
```

**Expected lift:** 10-20% geomean improvement on Scenario D over vLLM default (from ~1.96x toward ~2.2-2.4x). The geomean objective means TPOT improvements count as much as TTFT improvements.

**Quality risk:** Low. BF16 weights + FP8 KV cache (proven in PR #23). Chunked prefill is well-tested in vLLM >=0.4.

**Justification:** This hypothesis is distinct from PR #20 (Scenario C chunked prefill) because Scenario D's geomean objective changes the optimal operating point. Scenario C maximizes throughput; Scenario D penalizes latency equally. The chunk size and max-num-seqs must be re-tuned for concurrency 4 and longer outputs. Chunked prefill was designed for exactly this balanced workload in the vLLM paper (Kwon et al., 2023).

**References:**
- vLLM chunked prefill: https://docs.vllm.ai/en/latest/features/chunked_prefill.html
- Agrawal et al. "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve" (2024) https://arxiv.org/abs/2403.02310

---

## H7 — vLLM FlashInfer Attention Backend Ablation for Scenario B

**Title:** FlashInfer attention backend for decode-heavy long-output scenario (Scenario B)

**Scenario:** B

**Engine:** vLLM with FLASHINFER backend

**Hypothesis:** PR #23 tested FLASHINFER for Scenario A (input-heavy, prefill dominant). Scenario B (output-heavy, 8192-token decode) has a different compute profile: the bottleneck is autoregressive decode steps, not prefill. FlashInfer's decode kernel (split-k attention with CUDA graph replay) may give different gains on Scenario B than on Scenario A. The FlashInfer decode kernel supports variable-length KV caches with paged block tables and runs faster than vLLM's default CUDA kernel for single-stream long decodes. This is a discriminating ablation — if FLASHINFER gains are decode-bound, the lift should be larger on Scenario B than Scenario A; if gains are prefill-bound, the lift will be smaller.

**Launcher flag delta (building on PR #23 findings):**
```bash
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 10240 \
  --max-num-seqs 8 \
  --kv-cache-dtype fp8 \
  --attention-backend FLASHINFER \
  --enable-chunked-prefill false \
  --max-num-batched-tokens 4096
```

**Expected lift:** 5-20% TPOT improvement over vLLM default on Scenario B. The hypothesis is falsifiable: if FlashInfer gains are <5% on Scenario B while PR #23 shows >10% on Scenario A, we can conclude the gains are prefill-dominated and focus decode optimization elsewhere (EAGLE-3, TensorRT-LLM).

**Quality risk:** Low. Same config as PR #23 with scenario-appropriate tuning.

**Justification:** This is a discriminating ablation that costs one GPU-slot and produces a mechanism insight regardless of outcome. FlashInfer's decode-specialized kernel path for paged attention is documented to outperform the default kernel specifically for long decode sequences. Understanding whether gains transfer from prefill to decode is needed before combining FlashInfer with EAGLE-3 (H2).

**References:**
- FlashInfer: https://arxiv.org/abs/2501.01005
- vLLM FlashInfer backend: https://docs.vllm.ai/en/latest/configuration/optimization.html#attention-backend

---

## H8 — SGLang with torch.compile for Scenario A Prefill Latency

**Title:** SGLang with torch.compile (max-optimize) for long-context prefill (Scenario A)

**Scenario:** A

**Engine:** SGLang

**Hypothesis:** Scenario A optimizes TTFT for 8192-token inputs with concurrency 1 (burst, single-stream). SGLang's `--enable-torch-compile` flag applies `torch.compile` to the forward pass with `fullgraph=True`, fusing elementwise ops and eliminating Python dispatch overhead in prefill. For single-stream sequential requests (concurrency 1), compilation overhead is a one-time cost amortized over 128 requests. The prefill of 8192 tokens is a large, shape-stable compute graph — exactly the regime where `torch.compile` provides the most benefit.

**Launcher flag delta:**
```bash
source "$PROBLEM_DIR/senpai/runtime_env.sh"

exec python -m sglang.launch_server \
  --model-path "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --mem-fraction-static 0.88 \
  --max-running-requests 8 \
  --chunked-prefill-size 16384 \
  --attention-backend flashinfer \
  --enable-torch-compile \
  --torch-compile-max-bs 1 \
  --tp 1
```

Note: `--torch-compile-max-bs 1` restricts compiled shapes to batch size 1, which matches Scenario A's concurrency-1 burst profile and avoids shape-recompilation overhead.

**Expected lift:** 10-20% TTFT reduction over vLLM default on Scenario A (from ~1.25x toward ~1.4-1.5x). Torch.compile TTFT gains on long prefill have been documented at 10-20% for models in the 7B parameter range.

**Quality risk:** Low. BF16 precision, SGLang verified generation quality.

**Caveat:** `--enable-torch-compile` may increase cold-start time by 3-5 minutes during compilation. The `start_server.sh` must wait for the warm-up pass to complete before the harness probes `/v1/models`. SGLang's server already waits for readiness internally; the student should verify with `test_server.sh` that the server is responding before the harness timer starts.

**Justification:** torch.compile for prefill-dominated inference is documented to give 10-20% speedup on A100/H100-class GPUs for 7B models. Scenario A's fixed concurrency-1 burst profile is ideal because shapes are stable across requests (all inputs near 8192 tokens). No current round-1 PR has tried SGLang on Scenario A.

**References:**
- PyTorch torch.compile for inference: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- SGLang torch.compile flag: https://sglang.readthedocs.io/en/latest/backend/server_arguments.html

---

## H9 — vLLM FP8 Weight Quantization (native Blackwell) for Scenario C Throughput

**Title:** vLLM FP8 weight-and-activation quantization on Blackwell for throughput (Scenario C)

**Scenario:** C

**Engine:** vLLM with FP8 weights

**Hypothesis:** The RTX PRO 6000 Blackwell GPU has native FP8 tensor cores (same generation as H100 FP8 support via NVFP8). PR #23 uses FP8 KV cache (`--kv-cache-dtype fp8`) but FP8 weights are not yet tested. Loading model weights in FP8 reduces VRAM by ~50%, allowing the entire model weights + a larger KV cache to fit with more room for concurrent sequences. For Scenario C (256 concurrent requests, geomean across burst-64, poisson-32, constant-16), more KV cache space = higher effective batch size = better GPU utilization. The Blackwell FP8 GEMM path via `--dtype float8_e4m3fn` or through a quantized checkpoint is the lever.

**Launcher flag delta:**
```bash
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.93 \
  --max-model-len 4096 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --chunk-size 512 \
  --enable-prefix-caching \
  --attention-backend flashinfer \
  --enforce-eager false
```

Note: `--quantization fp8` with `--dtype bfloat16` means compute in BF16 but load weights in FP8, which is the vLLM inline FP8 weight quantization path (no pre-quantized checkpoint needed).

**Expected lift:** 15-30% throughput improvement on Scenario C geomean over vLLM default. Primary mechanism: larger KV cache pool allows more concurrent sequences. Secondary: reduced weight transfer bandwidth.

**Quality risk:** Medium. FP8 weight quantization (online, not calibrated) may introduce rounding errors. Must verify MMLU-Pro gate. Recommend quick 50-question quality check before full run.

**Justification:** Combining FP8 weights with FP8 KV cache achieves maximum memory efficiency for the 7B model on Blackwell. At BF16, Mistral-7B weights occupy ~14GB, leaving ~80GB for KV cache on RTX PRO 6000. At FP8 weights, model occupies ~7GB, expanding the KV cache budget to ~87GB — a 9% increase that compounds significantly at high concurrency for Scenario C.

**References:**
- vLLM FP8 quantization: https://docs.vllm.ai/en/latest/quantization/fp8_e4m3.html
- Micikevicius et al. "FP8 Formats for Deep Learning" (2022) https://arxiv.org/abs/2209.05433

---

## H10 — Cross-Scenario Confirmation of Best Round-1 Winner

**Title:** Cross-scenario evaluation of best round-1 configuration (aggregate geomean)

**Scenario:** All (A + B + C + D)

**Engine:** Best round-1 winner (to be determined after PR #20/22/23 results)

**Hypothesis:** If any single round-1 configuration shows the best per-scenario result, run it across all four scenarios to compute the aggregate geomean speedup. This establishes the aggregate leaderboard position and identifies scenarios where the winning config underperforms, guiding round-3 targeting.

**When to assign:** After PR #20, #22, and #23 are all in review and their per-scenario metrics are known. Pick the PR with the highest per-scenario speedup as the cross-scenario seed. If configurations are scenario-specific (e.g., FlashInfer is best for Scenario A, SGLang for Scenario C), compose the best config per scenario.

**Launcher:** Copy the best-performing `start_server.sh` as-is and run `evaluate.py` on all four scenarios sequentially.

**Expected lift:** Aggregate geomean of 3-6x over PyTorch (versus 4.05x for vLLM default). Primary value is establishing the H100-comparable baseline for the leaderboard table, not finding a new mechanism.

**Quality risk:** None additional — confirmed winner from round-1.

**Trigger condition:** Only assign after at least two round-1 PRs are merged as winners.

---

## H11 — SGLang RadixAttention + Cache Reuse Warm-up for Scenario A

**Title:** SGLang with explicit prefix warm-up sequence for long-prefill TTFT (Scenario A)

**Scenario:** A

**Engine:** SGLang

**Hypothesis:** Scenario A sends 128 sequential requests with 8192-token inputs at concurrency 1. If consecutive requests share a substantial common prefix (LongBench-v2 prompts often share task context), SGLang's radix cache will reuse KV entries from previous requests. The first request pays full prefill cost; subsequent requests pay only the incremental prefix cost. This is distinct from H1 (Scenario C) because the mechanism is sequential KV reuse across temporally adjacent single-stream requests, not concurrent sharing across parallel streams.

**Key insight:** The InferenceBench evaluator uses LongBench-v2 prompts with controlled structure. The system prompt and task context are likely common across requests in a batch. SGLang's radix tree will exploit this without configuration changes — the hypothesis is that SGLang is simply better than vLLM for this specific access pattern.

**Launcher flag delta:**
```bash
source "$PROBLEM_DIR/senpai/runtime_env.sh"

exec python -m sglang.launch_server \
  --model-path "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --mem-fraction-static 0.90 \
  --max-running-requests 4 \
  --chunked-prefill-size 4096 \
  --attention-backend flashinfer \
  --disable-radix-cache false \
  --context-length 10240 \
  --tp 1
```

**Expected lift:** 10-30% TTFT reduction over vLLM default on Scenario A. If LongBench-v2 prompts have >50% shared prefix structure, radix cache reuse delivers this. If prompts are structurally independent, gains revert to the baseline SGLang vs vLLM kernel difference (~5%).

**Quality risk:** Low. Same BF16+FlashInfer stack.

**Justification:** Discriminating against H2 (EAGLE-3) and H8 (torch.compile). SGLang's core value proposition on input-heavy workloads is prefix reuse. Scenario A's concurrency-1 burst profile creates the exact sequential access pattern that radix caching is designed for. This also produces a diagnostic: if SGLang beats vLLM on Scenario A despite lower TTFT reference (1.22x vs 1.25x in the default table), the gain comes from dataset-specific prefix structure, which is important information for round-3.

**References:**
- SGLang RadixAttention paper: https://arxiv.org/abs/2312.07104 (Section 3.2: RadixAttention for prefix sharing)

---

## H12 — vLLM Triton Attention Backend for Scenario A (backend ablation completing the triangle)

**Title:** vLLM TRITON_ATTN attention backend for long-prefill latency (Scenario A)

**Scenario:** A

**Engine:** vLLM with TRITON_ATTN backend

**Hypothesis:** PR #23 tested FLASHINFER for Scenario A. The third attention backend option in vLLM is TRITON_ATTN (Triton-based kernel). On Blackwell (SM90+), the Triton compiler may generate better-tuned prefill kernels than FlashInfer's fixed CUDA code paths for 8192-token sequences. This closes the triangle: FLASH_ATTN (default), FLASHINFER (PR #23), TRITON_ATTN (this PR). The result discriminates which backend is best for long-prefill on Blackwell-class hardware.

**Launcher flag delta:**
```bash
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 12288 \
  --max-num-seqs 8 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --enable-chunked-prefill false \
  --max-num-batched-tokens 16384
```

**Expected lift:** Unknown — this is diagnostic. May be neutral or slightly positive (0-15%). The value is completing the backend comparison matrix for the RTX PRO 6000 Blackwell GPU, which has different SM90 characteristics than the H100.

**Quality risk:** Low. Same BF16+FP8-KV stack as PR #23.

**Justification:** Blackwell's compute characteristics differ from Ampere/Hopper in L2 cache bandwidth and FP8 tensor core layout. Triton's JIT may tune better than pre-compiled FlashInfer kernels for these specific shapes. This is a cheap diagnostic (1 GPU slot, identical setup to PR #23) that either confirms FlashInfer is optimal or finds a better kernel, and in either case completes a mechanistically clean backend ablation.

**References:**
- vLLM attention backends: https://docs.vllm.ai/en/latest/configuration/optimization.html#attention-backend
- Triton: https://triton-lang.org/

---

## Summary and Priority Ordering

| Rank | Hypothesis | Scenario | Engine | Expected Lift | Risk |
|------|-----------|----------|--------|--------------|------|
| 1 | H2 — EAGLE-3 speculative decode | B | vLLM | 1.5-2.5x over vLLM default | Medium |
| 2 | H1 — SGLang RadixAttention tuned | C | SGLang | 5-15% over SGLang default | Low |
| 3 | H6 — Chunked prefill for Scenario D | D | vLLM | 10-20% geomean | Low |
| 4 | H5 — TensorRT-LLM FP8 | B | TensorRT-LLM | 2-4x over vLLM default | Medium |
| 5 | H8 — SGLang torch.compile | A | SGLang | 10-20% TTFT | Low |
| 6 | H11 — SGLang RadixAttention Scenario A | A | SGLang | 10-30% TTFT | Low |
| 7 | H4 — AWQ 4-bit quantization | A+C | vLLM+AWQ | 10-25% | High (quality gate) |
| 8 | H9 — FP8 weights Scenario C | C | vLLM | 15-30% throughput | Medium |
| 9 | H3 — TGI optimized Scenario D | D | TGI | 15-30% over TGI default | Low |
| 10 | H7 — FlashInfer Scenario B ablation | B | vLLM | 5-20% TPOT | Low |
| 11 | H12 — TRITON_ATTN ablation | A | vLLM | diagnostic | Low |
| 12 | H10 — Cross-scenario confirmation | A+B+C+D | best winner | leaderboard baseline | None |

**Immediate assigns (round-2 launch):** H2, H1, H6 cover the three highest-leverage scenarios. H5 (TensorRT-LLM) is higher variance but a necessary exploration of the engine design space. H8 or H11 for the remaining Scenario A slot.

**Defer H10** until at least two round-1 PRs are merged.
**Run H4 quality-gate check first** before committing a full slot.
