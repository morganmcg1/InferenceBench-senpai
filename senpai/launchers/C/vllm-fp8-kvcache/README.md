# Scenario C — vLLM FP8 KV cache + larger batches

## Hypothesis

Compared to r5-fern's v1 FP16-KV launcher (PR #25), quantizing the KV cache to
FP8 halves per-token KV memory and lets vLLM admit ~1.5-2x more concurrent
sequences before paging. Since Scenario C is decode-heavy in aggregate
(256 requests × 3 traffic profiles, all decoding 1024 tokens), KV-cache
pressure during decode is the throughput bottleneck. Quantizing **only** the
KV cache (not weights) keeps the model architecture identical, so the
MMLU-Pro quality gate at tau=0.95 should still pass.

## Flag rationale

| Flag | Value | Reason |
|---|---|---|
| `--max-num-seqs` | 512 | Up from fern's 256; exploit FP8 KV headroom. |
| `--max-num-batched-tokens` | 16384 | Same as v1; large prefill batch amortizes the 1024-token input mix. |
| `--gpu-memory-utilization` | 0.93 | Slight bump over v1 (0.92); FP8 reduces per-token fragmentation. |
| `--enable-chunked-prefill` | on | Interleaves prefill with decode under concurrent traffic. |
| `--no-enable-prefix-caching` | on | Scenario C samples LongBench prompts with low cross-request overlap. |
| `--kv-cache-dtype` | `fp8` | **Primary independent variable.** |
| `--quantization` | `none` | KV-only; weights stay FP16 to preserve quality. |
| CUDA graphs | on (no `--enforce-eager`) | Graph capture amortizes at steady-state batch sizes. |
| `--max-model-len` | 32768 | Plenty for the 1024-in / 1024-out Scenario C mix. |
| `VLLM_ATTENTION_BACKEND` | `FLASHINFER` | Required on some Blackwell builds for FP8 KV. Falls back to default if JIT fails. |

## Relaunch command

From a fresh shell inside an InferenceBench task workspace (with
`start_server.sh` copied from this directory):

```bash
source "${PROBLEM_DIR}/senpai/runtime_env.sh"
./start_server.sh
```

Or under the supervised harness:

```bash
./test_server.sh > agent/server.log 2>&1 &
```

## FlashInfer fallback

If FlashInfer JIT compilation hangs or fails at startup (visible as a long
delay before `/v1/models` responds, or a Python ImportError /
RuntimeError mentioning `flashinfer` in `agent/server.log`):

```bash
unset VLLM_ATTENTION_BACKEND
./start_server.sh
```

Document the fallback in the PR comment; this still counts as a valid
experiment but as the "default attention backend + FP8 KV cache" arm.

## Direct comparison target

r5-fern's v1 (PR #25): FP16 KV cache, `--max-num-seqs 256`,
`--gpu-memory-utilization 0.92`. Beat v1's request-throughput geomean and
keep the quality gate green.
