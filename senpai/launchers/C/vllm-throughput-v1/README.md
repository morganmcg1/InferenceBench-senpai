# Scenario C — vLLM throughput-tuned launcher v1 (FP16 KV cache)

PR #25 hypothesis: a throughput-tuned vLLM launcher with FP16 KV cache and
chunked-prefill / prefix-caching tuned for Scenario C's high-concurrency
short-prompt workload should land deep in the 30-50x range on this RTX PRO
6000 shakedown pod, in line with the H100 public reference snapshot
(`vLLM default` 48.69x; `SMAC3` 46.70x).

## Workload Recap (Scenario C)

- Input target 1024 tokens, output target 1024 tokens, 256 requests per profile.
- Three traffic profiles: `burst` c=64, `poisson` 32 req/s cap 32, `constant`
  16 req/s cap 16.
- Scenario C primary metric is the geomean of
  `request_throughput_req_per_s` across the three profiles.

## Flag Rationale

| Flag | Value | Why |
|---|---|---|
| `--max-num-seqs` | `256` | Match burst peak concurrency (64) with headroom; CUDA graphs can cover up to 256 batched sequences. |
| `--max-num-batched-tokens` | `16384` | Large prefill window so the 1024-input-token prefill amortizes well. |
| `--gpu-memory-utilization` | `0.92` | RTX PRO 6000 has ~96 GB VRAM; 0.92 leaves ~8 GB headroom for KV cache fragmentation and overhead. Starting point uses 0.90. |
| `--enable-chunked-prefill` | on | Interleave prefill with decode under concurrent traffic. |
| `--no-enable-prefix-caching` | off | LongBench prompts with `range_ratio=0.8` have low cross-request overlap; bookkeeping cost without a hit-rate. |
| (CUDA graphs) | on (default) | Omit `--enforce-eager`; graph capture amortizes at steady-state batch sizes. |
| `--kv-cache-dtype` | `auto` (FP16) | Preserves quality semantics. FP8 KV cache is r5-tanjiro's variant. |
| `--max-model-len` | `32768` | Plenty for 1024-in / 1024-out; keeps KV cache reasonable. |

The remaining flags (`--tokenizer-mode auto`, `--trust-remote-code`,
`--disable-log-stats`) match the starting-point launcher.

## Relaunch Recipe

```bash
cd "${PROBLEM_DIR}"
source senpai/runtime_env.sh

python senpai/create_task_workspace.py --scenario C \
  --output /tmp/inferencebench-scenario-c-fern --replace \
  --starting-point vllm_running \
  --launcher senpai/launchers/C/vllm-throughput-v1/start_server.sh

cd /tmp/inferencebench-scenario-c-fern/task
./test_server.sh > agent/server.log 2>&1 &
SERVER_PID=$!
# wait for /v1/models, then:
python evaluate.py --json-output-file metrics_full.json
kill -INT -- -"$(ps -o pgid= "$SERVER_PID" | tr -d ' ')" 2>/dev/null || kill "$SERVER_PID"
wait "$SERVER_PID" 2>/dev/null || true
```

Summarize and log with the SENPAI helpers in `senpai/`:

```bash
TORCH_BASELINE_JSON=$PROBLEM_DIR/src/eval/inference/baselines/speed/torch/inference_scenario_c_high_load/mistralai_Mistral-7B-Instruct-v0.3/baseline_metrics.json
python $PROBLEM_DIR/senpai/summarize_metrics.py metrics_full.json \
  --scenario C --baseline-metrics-json "$TORCH_BASELINE_JSON"
python $PROBLEM_DIR/senpai/log_metrics_to_wandb.py metrics_full.json \
  --scenario C --baseline-metrics-json "$TORCH_BASELINE_JSON" \
  --name "r5-fern/scenario-c-vllm-throughput-v1" \
  --group "scenario-c-vllm-shakedown"
```
