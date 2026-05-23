# InferenceBench SENPAI Target

Research target for open-ended LLM inference optimization. Given a fixed base
LLM, one benchmark GPU, a scenario-specific workload, and a wall-clock budget,
produce a reproducible OpenAI-compatible inference server that improves the
scenario primary metric while passing quality and integrity gates. The official
leaderboard setting is one NVIDIA H100 80GB GPU; current SENPAI shakedown runs
may use an NVIDIA RTX PRO 6000 Blackwell-class GPU with about 96GB VRAM and
must not be treated as leaderboard-comparable until repeated on H100.

This repository already contains the official InferenceBench benchmark harness.
The SENPAI layer in this target is an experiment coordination layer around that
harness, not a replacement for it. Normal experiment PRs should add or refine
launcher recipes, experiment notes, and helper tooling without changing the
benchmark evaluator.

## Mission

The research goal is to discover inference-serving configurations and systems
recipes that beat the strongest known benchmark references on each scenario.
The final deliverable of an experiment is not prose, a benchmark patch, or an
interactive server that happens to be alive. It is a reproducible launcher that
can be copied to task-local `./start_server.sh`, relaunched by the supervised
final-evaluation harness in a fresh container, and then evaluated by
`evaluate.py`.

The paper-facing metric is speedup over the naive PyTorch baseline. The raw
scenario objectives are higher-is-better after conversion, and speedup is
`candidate_raw_objective / pytorch_baseline_raw_objective`.

| Scenario | Workload focus | Raw objective used for speedup |
|---|---|---|
| A: Input-heavy | Long-context prefill latency | `1 / ttft.p50` from the `burst` profile |
| B: Output-heavy | Long decode latency | `1 / tpot.p50` from the `burst` profile |
| C: High-load | Concurrent throughput | Geomean of `request_throughput_req_per_s` across `burst`, `poisson`, and `constant` |
| D: General | Balanced serving | Geomean of `1/ttft.p50`, `1/tpot.p50`, and `request_throughput_req_per_s` from `burst` |

For individual experiment PRs, optimize one scenario at a time and report that
scenario's speedup over the PyTorch baseline as `SENPAI-RESULT.primary_metric`.
For mature winners, run occasional cross-scenario confirmation and report the
aggregate as the geometric mean of A-D speedups, matching the leaderboard.

Quality is a gate, not the ranking metric. A candidate that improves speed but
fails the MMLU-Pro quality gate is not a winner. A candidate that changes the
base model, evaluator, request set, or metric semantics is invalid even if the
numbers look good.

## Reference Snapshot

Timestamp: 2026-05-21. These are public reference numbers from the
InferenceBench README/site for Mistral-7B-Instruct-v0.3 with a 2 hour budget
per run on one NVIDIA H100 80GB GPU. This is the headline setting to target for
leaderboard claims unless the human research team explicitly launches a
different model, hardware, time budget, or starting-point ablation. RTX PRO 6000
shakedown results are useful for search direction and infrastructure hardening,
but should be repeated on H100 before claiming that they beat this table.
They are initial context for the advisor. The advisor branch should maintain
its own live `BASELINE.md` during a SENPAI run and compare terminal results
against that current state.

| Method | Aggregate | Sc. A TTFT | Sc. B TPOT | Sc. C req/s | Sc. D geomean |
|---|---:|---:|---:|---:|---:|
| SMAC3 search, 2h vLLM | 11.53x | 4.37x | 15.23x | 46.70x | 5.69x |
| TPE search, 2h vLLM | 11.25x | 4.48x | 14.76x | 43.46x | 5.58x |
| Random search, 2h vLLM | 10.20x | 4.21x | 11.34x | 41.81x | 5.42x |
| Best listed agent, Claude Sonnet 4.6 | 8.08x | 3.47x | 12.03x | 33.93x | 3.01x |
| vLLM default, no agent | 4.05x | 1.25x | 2.25x | 48.69x | 1.96x |
| SGLang default, no agent | 3.92x | 1.22x | 1.77x | 51.12x | 2.14x |
| HF TGI default, no agent | 3.30x | 1.14x | 1.37x | 41.94x | 1.80x |
| PyTorch baseline | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x |

## Codebase

- `README.md` - official benchmark overview, leaderboard, scenarios, gates, and
  submission quickstart. Read-only during normal experiment PRs.
- `src/eval/tasks/inference_scenario_*/scenario.json` - scenario workload
  definitions. Protected benchmark files.
- `src/eval/tasks/inference_scenario_*/mission.txt` and
  `workspace_conventions.txt` - task prompts and workspace rules used by the
  original agent harness. Read-only during normal experiment PRs.
- `src/eval/inference/runner.py` - official speed and quality evaluator.
  Protected. Read it to understand metrics, but do not edit it in normal
  experiment PRs.
- `src/eval/inference/quality_gate.py` - MMLU-Pro quality gate. Protected.
- `src/eval/inference/hpo_search_baselines.py` - non-agent search baseline and
  useful reference for objective computation, server launch, and search spaces.
  Read-only unless the advisor explicitly assigns benchmark tooling work.
- `src/baselines/search_spaces/*.yaml` - HPO search spaces for vLLM, SGLang,
  and TGI. Read-only reference for normal launcher experiments.
- `src/starting_points/vllm_running/start_server.sh` - default vLLM starting
  point used by the original harness. Treat as read-only for normal SENPAI
  experiments unless the advisor explicitly assigns an integration PR.
- `src/eval/tasks/_shared/task_context/start_server.sh` - blank task-local
  launcher stub copied into benchmark jobs. Protected template.
- `agents/*/solve.sh`, `src/run_task.sh`, `src/commit_utils/*`, `containers/*`
  - benchmark orchestration and container plumbing. Protected for normal
  experiment PRs.
- `program.md`, `instructions/prompt-advisor.md`,
  `instructions/prompt-student.md` - SENPAI target contract and role overlays.
  Read-only during normal experiment PRs.
- `senpai/` - SENPAI-only helper area for launcher recipes, result summarizers,
  notes, and future integration scripts. This is the preferred place for new
  target-package functionality.

Normal student PRs should avoid touching existing benchmark code. Put reusable
launcher recipes under `senpai/launchers/<scenario>/<slug>/start_server.sh`,
analysis under `senpai/research/` or the advisor-managed `/research/`
directory, and helper scripts under `senpai/`.

Do not add a local shortcut that changes the benchmark semantics. If students
run from SENPAI GPU pods rather than an official HTCondor job, they should still
use the same task files, `start_server.sh` foreground launcher contract,
`evaluate.py` metrics, precomputed request files when available, quality gate,
and clean supervised relaunch behavior that the official harness uses.

## Benchmark Workloads

The evaluator samples LongBench-v2 prompts with controlled token lengths. Speed
requests are not scored for semantic accuracy; semantic preservation is checked
separately by the quality gate.

| Scenario | Input target | Output target | Requests | Traffic profile |
|---|---:|---:|---:|---|
| A | 8192 tokens | 1024 tokens | 128 | `burst`, concurrency 1 |
| B | 1024 tokens | 8192 tokens | 64 | `burst`, concurrency 1 |
| C | 1024 tokens | 1024 tokens | 256 each | `burst` concurrency 64, `poisson` 32 req/s cap 32, `constant` 16 req/s cap 16 |
| D | 4096 tokens | 2048 tokens | 96 | `burst`, concurrency 4 |

Per-request input and output lengths are drawn uniformly from 80 percent to 100
percent of the target length. Scenario C reports the geomean throughput across
its three traffic profiles.

## Server Contract

Every candidate must expose OpenAI-compatible endpoints:

- `GET /v1/models`
- `POST /v1/chat/completions`
- Streaming responses for chat completions

The final launcher must run in the foreground and bind to the provided host and
port:

```bash
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MODEL_ID="${INFERENCE_BENCH_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
```

Use `exec ...` for the final server process. Do not daemonize with `nohup`,
`setsid`, or a trailing `&`. Do not hardcode the port. The official benchmark
kills the live server and relaunches `start_server.sh` in a clean supervised
container before final scoring.

Allowed serving levers include:

- Engine choice: vLLM, SGLang, TGI, TensorRT-LLM, or a custom OpenAI-compatible
  server.
- Runtime tuning: KV-cache dtype and allocation, max sequences, batched token
  limits, chunked prefill, prefix caching, scheduler policy, CUDA graphs,
  attention backend, tokenizer mode, and memory utilization.
- Quantization: FP8, GPTQ, AWQ, bitsandbytes, or other formats, only if the
  same base model architecture is preserved and the quality gate still passes.
- Kernel and backend tuning: FlashAttention, FlashInfer, Triton kernels,
  speculative decoding, CUDA/Triton cache placement, and compile settings.

These categories are examples, not a whitelist. Creative approaches are allowed
when they preserve the assigned model, OpenAI-compatible API, metric semantics,
quality gate, integrity rules, and clean relaunch behavior.

The base model must remain the assigned `INFERENCE_BENCH_BASE_MODEL`. A
different model repo, smaller substitute, external API, response cache, or
pre-generated output invalidates the run.

## Launch Preflight And Scoring Assets

Before assigning serving experiments in a paper-grade run, the advisor should
verify that scoring assets already exist for the active hardware, model, seeds,
and scenarios:

```bash
# Current RTX PRO 6000 shakedown mode
python senpai/preflight.py --scenario all \
  --expected-gpu "RTX PRO 6000" \
  --require-wandb

# Later H100 leaderboard-comparable mode
python senpai/preflight.py --leaderboard-mode --scenario all \
  --expected-gpu H100 \
  --require-wandb
```

This check must pass before claiming paper-grade results for the active
hardware, and the `--leaderboard-mode --expected-gpu H100` version must pass
before claiming leaderboard-comparable results. It verifies the GPU class, W&B
availability, deterministic speed request files, PyTorch speed baseline
metrics, MMLU-Pro quality samples, and the PyTorch quality baseline registry.
If it fails, treat baseline/request setup as the first research task; do not let
students fabricate `speedup_over_pytorch` from public H100 ratios or report raw
objectives as if they were speedups.

For Kubernetes SENPAI launches, arm `senpai/arm_cluster_cutoff.sh` at startup
instead of using a bare cleanup job. For a strict 2 hour window, pass the same
PVC `--start-gate-path` to the cutoff script and `k8s/launch.py`; pods wait at
the gate until every expected pod is Ready, then the cutoff job opens the gate
and starts the clock. It archives `/root/.claude` from every tagged
advisor/student pod to the PVC shortly before shutdown, and then deletes the
tagged SENPAI deployments/configmaps/secrets.

If request generation is blocked by tokenizer/runtime drift, materialize
deterministic request files once from the SENPAI helper instead of editing
protected evaluator code:

```bash
INFERENCE_BENCH_ALLOW_HF_DOWNLOAD=1 \
python senpai/materialize_requests.py --scenario all --backend torch \
  --base-model mistralai/Mistral-7B-Instruct-v0.3 --seed 248
```

For pod-local work, create task workspaces that mirror the official task
layout rather than improvising from the repository root:

```bash
python senpai/create_task_workspace.py --scenario A \
  --output /tmp/inferencebench-scenario-a \
  --starting-point vllm_running
```

The task workspace still uses the official `evaluate.py`, quality gate, request
files, launcher contract, and supervised relaunch shape. The helper only stages
those pieces in a predictable pod-local directory.

For RTX PRO 6000 or other shared-pod shakedown runs, source the SENPAI runtime
environment helper before starting vLLM/SGLang/TGI:

```bash
source "$PROBLEM_DIR/senpai/runtime_env.sh"
```

This sets CUDA pip-package include/library paths and pod-local JIT caches that
avoid common Blackwell/FlashInfer/vLLM startup failures. It is launch
environment setup only; it does not modify the evaluator or scoring contract.

## Running

For local or pod-level exploration, start from a launcher recipe and copy it
into the active task workspace as `./start_server.sh`. The original benchmark
task workspace contains `evaluate.py`, `test_server.sh`, `timer.sh`,
`scenario.json`, and the task-local launcher.

Inside an active InferenceBench task workspace:

```bash
./test_server.sh > agent/server.log 2>&1 &
python evaluate.py --quick --json-output-file metrics_quick.json
python evaluate.py --json-output-file metrics_full.json
python "$PROBLEM_DIR/senpai/summarize_metrics.py" metrics_full.json \
  --scenario A \
  --baseline-metrics-json /path/to/pytorch_baseline_metrics.json
python "$PROBLEM_DIR/senpai/log_metrics_to_wandb.py" metrics_full.json \
  --scenario A \
  --baseline-metrics-json /path/to/pytorch_baseline_metrics.json \
  --name "$STUDENT_NAME/<short-description>" \
  --group "<hypothesis-or-pr>"
```

When running from the repository root for static checks or helper development:

```bash
python senpai/summarize_metrics.py path/to/metrics.json --scenario C \
  --baseline-metrics-json path/to/pytorch_baseline_metrics.json
python -m pytest tests/test_hpo_search_baselines.py tests/test_request_sampling.py
```

Use quick evaluation for viability and full evaluation before terminal results.
Quick runs are useful for launch failures and large regressions, but they are
not sufficient evidence to merge a winner. A terminal winner needs a clean
relaunch and full `evaluate.py --json-output-file ...` result unless the advisor
explicitly marked the PR as tooling-only.

## Metrics And Telemetry

The official evaluator writes `metrics.json` with:

- `profiles.<profile>.ttft.p50`, `p90`, `p99`
- `profiles.<profile>.tpot.p50`, `p90`, `p99`
- `profiles.<profile>.itl.p50`, `p90`, `p99`
- `profiles.<profile>.request_throughput_req_per_s`
- `profiles.<profile>.generation_throughput_tokens_per_s`
- `profiles.<profile>.success_count`, `failure_count`, and `failure_rate`
- `quality_check.pass`
- `quality_check.datasets.mmlu_pro.observed_accuracy`
- `quality_check.datasets.mmlu_pro.baseline_accuracy`
- `quality_check.datasets.mmlu_pro.ratio`

The primary ranking metric is scenario-dependent and must be computed from the
full clean-relaunch metrics. Tail latency, failure rate, empty outputs, VRAM
peak, and server logs are required diagnostics when deciding whether a result is
really robust.

W&B logging is required for SENPAI runs. Log to:

```text
wandb-applied-ai-team/inferencebench-senpai
```

Use `senpai/log_metrics_to_wandb.py` after each meaningful quick or full eval.
It logs the raw metrics, scenario raw objective, paper-facing speedup when a
PyTorch baseline is supplied, quality gate fields, and the metrics artifact.
The baseline metrics file or `--baseline-primary` value must come from the
matching PyTorch baseline for the same scenario and benchmark setup.
The official benchmark harness is still the source of truth for evaluation; the
W&B bridge is a read-only logger around its output.

## Quality And Integrity Gates

The quality gate evaluates a fixed 500-question MMLU-Pro subset with greedy
decoding and requires observed accuracy to be at least `tau = 0.95` times the
PyTorch baseline accuracy. The evaluator reports the dataset-level observed
accuracy, baseline accuracy, ratio, and pass/fail status.

The integrity gate is benchmark-owned and considers the transcript and final
launcher. The following are forbidden:

- Editing `evaluate.py`, `runner.py`, `quality_gate.py`, scenario files,
  baseline request files, generated request logs, or final metrics files to
  change the score.
- Replacing the assigned base model with a different model, smaller model,
  distilled model, or unrelated community checkpoint. Quantized versions are
  only acceptable when they are legitimate quantizations of the assigned base
  model and pass quality.
- Offloading inference to external APIs or remote services.
- Caching, memorizing, or intercepting benchmark requests or quality-gate
  answers.
- Returning synthetic or pre-generated text instead of model generations.
- Depending on an already-running daemon, interactive shell state, or files that
  are not recreated by the final launcher in a fresh container.
- Hardcoding the evaluation host, port, scenario seed, request IDs, or paths in
  a way that only works during the student's session.

If an evaluator bug is suspected, ask the advisor first and separate that work
from serving-optimization PRs.

## Experiment Strategy

This target gives agents broad freedom over framework, optimization, and
parameter choices. SENPAI adds coordination: the advisor manages scarce time and
GPU access, students run bounded research arms or single hypotheses, and every
decision is measured through the official evaluator.

Do not overfit to this document's example levers. Choose strategies based on the
active scenario, current live baseline, measured failures, remaining wall-clock
time, and available GPU capacity.

## Results Contract

Student result comments must include a single-line marker:

```markdown
SENPAI-RESULT: {"terminal":true,"status":"complete","pending_arms":false,"wandb_run_ids":["<run-id>"],"primary_metric":{"name":"scenario/A/speedup_over_pytorch","value":0.0},"test_metric":{"name":"quality/mmlu_pro_observed_accuracy","value":0.0}}
```

Use the appropriate primary metric name:

- Scenario A: `scenario/A/speedup_over_pytorch`
- Scenario B: `scenario/B/speedup_over_pytorch`
- Scenario C: `scenario/C/speedup_over_pytorch`
- Scenario D: `scenario/D/speedup_over_pytorch`
- Cross-scenario confirmation: `aggregate/geomean_speedup_over_pytorch`

Also report:

- Scenario, base model, GPU type, and time budget.
- Exact launcher path and the exact `start_server.sh` contents or diff.
- Exact evaluation command and whether it was quick or full.
- `metrics.json` path or attached artifact.
- Quality gate status and MMLU-Pro observed/baseline/ratio.
- Paper-facing speedup over PyTorch, raw primary objective, baseline raw
  objective, plus TTFT, TPOT, ITL, request throughput, generation throughput,
  success/failure counts, and VRAM peak.
- Comparison against the advisor-maintained current baseline and the reference
  snapshot above when relevant.
- W&B run ID from `wandb-applied-ai-team/inferencebench-senpai`.
- Server start time, cold-relaunch result, and any launch caveats.
- What happened, why the hypothesis did or did not work, and suggested
  follow-ups.

Set `terminal=true` only when every advisor-required arm is complete and no
pending run can change the conclusion. Negative results are useful when they
clearly identify the failed hypothesis, failure mode, and next likely move.

## Advisor Guidance

Detailed advisor workflow lives in `instructions/prompt-advisor.md`. This
program defines the metric, contract, protected boundaries, and result schema;
the advisor should choose concrete strategy based on live evidence from the
current run.

## Roles

Research is coordinated through GitHub PRs with an advisor/student model.
GitHub Issues are used for communication with the human research team. See
`instructions/prompt-advisor.md` and `instructions/prompt-student.md`.
