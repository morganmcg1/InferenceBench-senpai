# Auto GPU Kernel Competition Lessons For SENPAI

Sources reviewed:

- [Dogacel/auto-gpu-kernel](https://github.com/Dogacel/auto-gpu-kernel), commit
  `126135c` as cloned on 2026-05-25.
- [Auto GPU Kernel report.pdf](https://github.com/Dogacel/auto-gpu-kernel/blob/main/report.pdf).
- The repository's winning DSA track task directories, especially their
  `CLAUDE.md`, `.claude/commands/`, `experiments/summary.md`,
  `experiments/LESSONS.md`, `experiments/workload_profile.md`, and
  `experiments/profile.md` files.

This note is for InferenceBench SENPAI advisors and students. It is not a
request to turn InferenceBench into a custom kernel contest. The valuable part
is the operating system: how the winning agent measured, searched, preserved
knowledge, and converted GPU systems judgment into reliable improvements under
competition pressure.

## Executive Summary

The winning auto-gpu-kernel system placed first on the MLSys 2026 FlashInfer AI
Kernel Generation Contest DSA track with an average reported speedup of 34.93x.
Its final submissions were two Triton kernels:

| Task | Final reported latency |
|---|---:|
| Sparse attention, `h16_ckv512_kpe64_topk2048_ps64` | 0.010 ms |
| TopK indexer, `fp8_h64_d128_topk2048_ps64` | 0.016 ms |

The decisive advantages were not a single magic kernel trick. The winner built
a disciplined autonomous optimization loop:

- Inspect the actual workload before tuning the implementation.
- Make one meaningful change at a time unless a bounded matrix is the
  hypothesis.
- Use cheap correctness and stride benchmarks for iteration, then reserve full
  benchmarks for confirmed candidates.
- Compare small deltas with paired A/B runs in the same VM or pod.
- Log every experiment, including failures, in durable files that future agents
  must read.
- Preserve semantic constraints even when exploiting data distribution,
  padding, or shape regularities.
- Prefer abstractions that the agent can edit and debug reliably over lower
  level tools that look theoretically faster but cause syntax and correctness
  churn.

For InferenceBench, the translation is direct: the advisor should be a research
director and scorekeeper, students should be measured searchers, W&B and PR
comments should be the durable memory, and no one should spend a two-hour run
wandering through unmeasured ideas.

## What Actually Won

The report describes a full-agent submission: Claude Code controlled the loop,
edited kernels, benchmarked them, and logged results without human intervention.
The successful system was built around three durable artifacts:

- `CLAUDE.md`: invariant rules, benchmark commands, correctness hazards, and
  the current optimization target.
- `experiments/summary.md`: a chronological ledger of attempted changes,
  measured results, decisions, and next directions.
- `experiments/LESSONS.md`: distilled negative and positive lessons that the
  agent had to consult before repeating an idea.

The winner also used small, purpose-built slash commands:

- `/optimize`: read the current state, choose one change, implement, validate,
  benchmark, log, and decide keep or revert.
- `/benchmark`: run the configured quick, stride, or full benchmark.
- `/log-experiment`: archive the exact kernel, benchmark log, and result note.

The key pattern is that the agent's memory lived in files, not in a long chat
transcript. This matters for SENPAI because advisors and students are separate
processes. If the only state is conversational, coordination decays quickly.
The InferenceBench analog should be `BASELINE.md`, W&B runs, PR comments with
terminal `SENPAI-RESULT` markers, launcher recipes, and short research notes
under `senpai/research/`.

## Winning Loop

The winning loop was deliberately boring:

1. Re-read the current kernel, baseline semantics, experiment summary, lessons,
   and the newest incomplete plan.
2. Pick exactly one optimization or one bounded matrix.
3. Explain why this differs from prior failures.
4. Implement the change.
5. Run a cheap correctness check.
6. Run the standard stride benchmark.
7. If the delta is small, run paired A/B on the same machine.
8. Log the result whether it wins, loses, crashes, or fails correctness.
9. Keep the change only when the evidence justifies it.

The report emphasizes that shallow search was the enemy. The winning agent did
not need perfect first ideas. It needed a loop that prevented idea drift,
measurement drift, and repeated mistakes.

For InferenceBench, SENPAI should treat each student PR as either:

- A single launcher hypothesis, such as a specific vLLM scheduler or KV-cache
  recipe.
- A bounded research arm, such as a small matrix over engine, precision, and
  concurrency settings with an explicit stop rule.

The advisor should make that distinction explicit. A bounded arm is not vague
permission to wander; it is a compact search space, a scenario, a metric, a
time budget, and a required final full evaluation for the best valid candidate.

## Benchmarking Discipline

The competition winner used tiered benchmarking:

| Mode | Purpose |
|---|---|
| Quick | Correctness and obvious breakage on a tiny workload subset. |
| Stride | Default per-iteration measurement on a representative slice. |
| Full | Final confirmation for candidates that already look good. |

This is especially important for InferenceBench. Full scenario evaluation is
expensive, and single-GPU SENPAI launches have a hard two-hour wall clock. The
advisor should require students to use cheap probes to reject bad ideas, but
never accept a winner without the official evaluator and clean relaunch.

The kernel contest also found that reference latency drifted by roughly 20 to
30 percent across VMs, making raw speedups against a stale reference unreliable
for small claims. Their solution was to track absolute latency during a run and
use paired same-machine A/B comparisons for sub-5 to sub-10 percent deltas.

InferenceBench's paper-facing metric is still speedup over PyTorch baseline,
because that is the benchmark contract. But inside a live SENPAI run:

- Compare candidates against the advisor's current live baseline on the same
  hardware and scoring assets.
- Treat tiny speedup differences as noise until confirmed by a paired rerun.
- Preserve raw TTFT, TPOT, throughput, failure rate, and quality details in
  W&B so the advisor can diagnose why a candidate moved the headline metric.
- Avoid public H100 numbers as a denominator for RTX PRO 6000 shakedowns.

## Subagents And Their SENPAI Analog

The auto-gpu-kernel setup used three specialized helper agents.

| Source role | What it did | InferenceBench SENPAI analog |
|---|---|---|
| Workload inspector | Studied actual inputs, padding, shapes, locality, and regime splits. | A student or advisor pass over `scenario.json`, request lengths, traffic profile, model length, concurrency, and per-request metrics before assigning knobs. |
| Profiler | Attributed time to phases and wrote a durable `profile.md`. | vLLM/SGLang/TGI logs, W&B metrics, GPU utilization, startup time, TTFT/TPOT/throughput split, and optional Nsight/CUPTI when useful. |
| Research agent | Clean-context pivot after plateau or repeated failures. | Advisor reassignment or a fresh student PR that reads only the durable artifacts and proposes the next search tier. |

The most important lesson is that workload inspection was not optional. The
report says the base agent tended to over-focus on implementation. The winning
system forced it to examine the distribution of inputs and discovered large
special cases there.

InferenceBench has the same trap. It is tempting to tune `--max-num-seqs`,
`--max-num-batched-tokens`, or engine defaults blindly. The better loop starts
from the scenario:

- Scenario A is long-prefill, low-concurrency. TTFT dominates.
- Scenario B is long-decode, low-concurrency. TPOT dominates.
- Scenario C is concurrent serving throughput under several traffic profiles.
- Scenario D mixes prefill, decode, and moderate concurrency.

The advisor should require each serious PR to say which part of the serving
timeline it expects to improve and how the evaluator will reveal that movement.

## Measured Milestone Map

The experiment logs are useful because they show how the winning results
emerged. They were not first-shot ideas. They were a sequence of structural
changes, targeted reverts, and retuning after each major rewrite.

### TopK indexer path

| Stage | Approximate reported result | Lesson |
|---|---:|---|
| Initial fused Triton score kernel | 2.408 ms | A correct custom kernel can still be far from the right algorithm. |
| Direct FP8 tensor-core dot | 2.179 ms | Hardware-aware math helped, but only modestly. |
| Batched `torch.topk` plus gather | 0.310 ms | Replacing the wrong custom structure with a better library primitive was a huge early win. |
| Zero-copy FP8 cache view | 0.200 ms | Removing layout copies beat more arithmetic tuning. |
| Fused remap/post-processing kernel | 0.049 ms | Many small framework ops were the true bottleneck after the first rewrite. |
| Scoreless path for short contexts | 0.0276 ms full mean in one logged candidate | Workload-specialized semantics can remove whole computation phases. |
| Custom Triton radix-select | 0.0199 ms stride result | Once `torch.topk` became the bottleneck, a task-specific selector paid off. |
| `num_warps=8` for reduction-heavy radix path | 0.0116 ms stride result | Tuning worked after the algorithm was right, not before. |
| Later block-pointer and prefix packing work | Marginal improvements | Small low-level wins required careful paired measurement. |

Several failed TopK ideas were just as important: GPU-derived host branching
caused synchronization, larger page-per-program groupings increased register
pressure, generic `tl.sort` scaled poorly beyond the right block size, and
Python buffer caches did not remove the underlying cost.

### Sparse attention path

| Stage | Approximate reported result | Lesson |
|---|---:|---|
| Initial fused Triton kernel | 0.100 ms | Correct fusion was a baseline, not the finish line. |
| Split-K style parallelism plus combine | 0.024 ms | The workload had too little token-level parallelism, so the kernel needed a new parallel axis. |
| Dynamic loop bounds and D-dimension combine parallelism | 0.020 to 0.022 ms | Padding-aware work reduction helped when it reduced live memory traffic. |
| `BLOCK_N=128` and related retuning | 0.016 ms | Tile tuning mattered after the split/combine structure existed. |
| Hybrid dispatch by token count | Similar median, better small-token strata | One universal kernel was not best for every workload regime. |
| Single-launch split+combine with in-kernel barrier | New best in the logs | Launch overhead and combine overhead became worth fusing only after earlier wins. |
| Volatile polling, monotonic counters, cache hints | Marginal improvements | Synchronization/cache details helped after the main structure was right. |
| Retuning split count from 8 to 16 after rewrites | Large win for some token-count strata | Old tuning optima expired after structural changes. |

The Sparse Attention logs also show a common trap: a global workload statistic
misled the research agent until it analyzed the actual dispatched stride
workloads. For InferenceBench, a summary average can hide the traffic profile
or request subset that determines the score.

## Domain Lessons From The TopK Kernel

The TopK task optimized an indexer over an FP8 cache. The largest wins came
from changing the computation, not just twiddling launch parameters.

Useful lessons:

- Distribution-aware fast paths can beat a general algorithm. Many workloads
  had rows shorter than the top-k threshold, so the kernel could return natural
  order indices without scoring or sorting while preserving the set-based
  correctness rule.
- Zero-copy layout interpretation mattered. The winning path used strided views
  over the actual FP8 cache layout instead of copying or making contiguous
  tensors.
- Algebraic movement mattered. Per-token scale factors could be applied after
  a nonnegative reduction instead of inside the dot-product inner loop.
- Fusing the remap/post-processing path removed many small PyTorch operations
  and became a major latency reduction.
- Replacing `torch.topk` with a custom Triton radix-select was a structural
  win, because generic sort/top-k behavior became the bottleneck.
- Warp count mattered only when the kernel was reduction-heavy. `num_warps=8`
  helped for large block reductions but was not a universal tuning win.
- Host synchronization was poisonous. Any approach that used GPU-derived
  values through `.item()` or host branching lost badly.
- Padding and inactive work only matter if they are on the critical path.
  Removing work from lanes or rows that were already hidden did not improve
  wall-clock latency.
- Python allocation caching was mostly a distraction because PyTorch already
  caches device memory; the real gain would have been eliminating allocation
  calls or preserving a downstream layout.

InferenceBench translation:

- Look for scenario-specific fast paths that preserve the benchmark contract:
  not semantic cheating, but legitimate serving specialization.
- Avoid extra tensor copies, tokenizer conversions, file I/O, or Python-side
  post-processing in the request path.
- Prefer changes that remove a whole phase or synchronization point over
  changes that slightly tune a phase not on the critical path.
- Be suspicious of knobs that look good in isolation but increase memory
  pressure, compilation churn, or launch overhead.
- If a metric moves by a tiny amount, rerun it in a paired setup before the
  advisor updates the live baseline.

## Domain Lessons From The Sparse Attention Kernel

The Sparse Attention task rewarded phase decomposition and then launch
reduction. The early best design used split-K style parallelism plus a combine
step. Later wins fused split and combine into one launch with an in-kernel
barrier, then retuned split counts after structural changes.

Useful lessons:

- Small token counts created low occupancy. Parallelizing over TopK positions
  and splitting work across the sparse dimension exposed more parallel work.
- Valid entries were often a contiguous prefix with heavy padding. Algorithms
  that respected that distribution avoided wasted memory traffic.
- Hybrid dispatch by token count beat one universal path: tiny-token cases
  preferred a fused/simple path, larger cases preferred split+combine.
- Online softmax in base-2 form matched expected output semantics and avoided
  unnecessary conversions.
- In-kernel synchronization was useful only after the algorithm shape made
  combine overhead a true bottleneck.
- Atomic and polling details mattered, but only at the margin after the
  structural shape was right.
- Retuning `NUM_SPLITS` after structural rewrites produced a large win. Old
  tuning optima did not remain valid after the kernel changed.
- Cache hints such as evict-first or evict-last were small and measurement
  sensitive; they required per-stratum analysis.
- Some "obvious" low-level ideas regressed: bf16 partial accumulators, larger
  blocks, different stage counts, and adaptive host dispatch all failed in
  context.

InferenceBench translation:

- Do not assume the best engine/knob for one scenario remains best after
  changing precision, max model length, traffic profile, or scheduler settings.
- Scenario C is especially sensitive to scheduler and concurrency structure.
  A change that improves one traffic profile can regress another.
- If a candidate changes the serving architecture, retune the nearby knobs
  instead of reusing the previous defaults.
- Treat startup, prefill, decode, queueing, and quality as separate phases.
  Improvements compound when the current bottleneck is correctly identified.

## Abstraction Choice: Why Triton Won There

The report is nuanced about tools. CUDA and CuTeDSL may have a higher ceiling,
but the autonomous loop lost too much time to syntax, compile errors, and
correctness churn. Triton won because the agent could edit it reliably, test it
quickly, and keep momentum.

For InferenceBench, the same principle applies:

- vLLM, SGLang, TGI, TensorRT-LLM, custom OpenAI-compatible servers, and
  hand-written kernels are all possible levers.
- The best first choice is not necessarily the theoretical maximum. It is the
  tool that gives the current student enough control, observability, and
  iteration speed within the two-hour budget.
- A tool switch is expensive. Use it when the current path has plateaued or
  when workload evidence points at a capability the current engine cannot
  expose.
- A lower-level path should earn its cost by removing a real bottleneck, not by
  sounding more advanced.

The contest also found model capability mattered. The winning loop depended on
an agent strong enough to maintain syntax, correctness, and measurement
discipline. For SENPAI, prompts and tooling should reduce cognitive load:
clear commands, stable paths, preflight checks, W&B logging, and explicit GPU
coordination are not luxuries.

## Anti-Patterns To Avoid

The report and experiment logs repeatedly punished these patterns:

- Repeating a failed idea because the reason for failure was not logged.
- Trusting in-context memory instead of re-reading durable artifacts.
- Making broad changes, then being unable to tell which part helped.
- Running full benchmarks for obviously broken candidates.
- Accepting quick-benchmark wins without full validation.
- Measuring against stale references across machines.
- Optimizing a phase that profiling shows is not on the critical path.
- Adding host synchronizations to make an adaptive path easier.
- Switching implementation languages or engines too early in the search.
- Mistaking allocation caching or wrapper changes for real request-path wins.
- Treating small cache-hint or tuning deltas as real without paired reruns.
- Ignoring failed experiments; failures were often the fastest route away from
  bad search regions.

In InferenceBench terms, the advisor should call out:

- Missing or stale `BASELINE.md`.
- Results with no W&B run or no full metrics artifact.
- Launcher recipes that depend on a live process, shell history, warmed cache,
  or untracked file.
- PRs that report only "it was faster" without TTFT, TPOT, throughput, failure
  rate, and quality context.
- Students waiting for perfect instructions instead of running bounded, useful
  probes inside the assigned surface.

## Advisor Checklist

Before assigning work:

- Verify scoring preflight for the active hardware and scenario.
- Read the current live `BASELINE.md` and recent W&B runs.
- Decide whether the PR is one hypothesis or a bounded research arm.
- State scenario, primary metric, baseline, expected phase movement, GPU queue
  expectations, quick-eval budget, stop rule, and final full-eval requirement.
- Assign at least one workload-inspection or log-analysis task when the search
  is drifting into blind knob tuning.

During the run:

- Keep score visibly. Update the live baseline when a clean candidate improves
  the current best.
- Check active PRs frequently enough that two-hour time does not disappear in
  stalls.
- Let students run local arms when communication overhead would dominate, but
  require them to stay inside the assigned surface.
- Route scarce GPU time to the experiment most likely to answer a live
  decision.
- Use low-GPU tasks for other students: log review, launcher prep, engine
  install checks, request-shape analysis, W&B comparison, and next-arm planning.
- For small claimed wins, ask for paired confirmation before merging or
  updating the live baseline.

After a result:

- Merge only reproducible, quality-passing, clean-relaunch improvements.
- Send promising non-winners back with a specific next variant.
- Close dead ends with the lesson captured so future students do not repeat
  them.
- If progress stalls, change the search tier: engine, precision, scheduler,
  concurrency, caching, startup, or serving architecture.

## Student Checklist

Before editing:

- Read `program.md`, the advisor PR assignment, current baseline notes, and
  any relevant `senpai/research/` notes.
- Inspect the actual scenario workload and request shape.
- Identify the serving phase you are trying to improve.
- Decide the minimum quick check that can reject the idea.

While experimenting:

- Change one thing at a time unless assigned a bounded matrix.
- Keep launcher changes reproducible in files, not in shell history.
- Log W&B metrics for every valid measurement.
- Record failures and infra issues distinctly from true performance losses.
- Do not run heavyweight GPU work without respecting the advisor's GPU
  coordination plan.
- Avoid protected benchmark changes unless explicitly assigned tooling work.

Before reporting:

- Run the official evaluator on the best candidate.
- Include the exact launcher recipe.
- Include raw metrics, speedup, quality status, failure rate, hardware, and
  W&B run ID.
- Explain which phase moved and why you believe the change caused it.
- State whether any arms remain pending.

## Concrete InferenceBench Ideas Inspired By The Report

These are examples, not a whitelist. The right idea depends on the active
scenario, hardware, and live baseline.

- Scenario A: prefill-oriented tuning, chunked prefill decisions, attention
  backend, prefix cache behavior, max batched tokens, CUDA graph capture
  behavior, and model length limits.
- Scenario B: decode-oriented tuning, KV-cache dtype/layout, speculative
  decoding where quality and contract permit, scheduler limits, CUDA graphs,
  and long-output memory pressure.
- Scenario C: concurrency and scheduler experiments, max running requests,
  queueing behavior, traffic-profile tradeoffs, engine comparison, and failure
  rate under sustained load.
- Scenario D: balanced recipes that avoid winning TTFT while losing TPOT or
  throughput enough to hurt the geomean.
- Cross-engine: compare vLLM, SGLang, TGI, TensorRT-LLM, or a custom
  OpenAI-compatible server when the current engine cannot expose the needed
  control or has plateaued.
- Runtime hygiene: cold relaunch, dependency isolation, cache placement,
  tokenizer setup, startup time, port cleanup, and GPU process cleanup.

The report's deeper lesson is to prefer mechanisms over labels. "Try FP8" is
not a hypothesis. "FP8 KV-cache should reduce decode memory bandwidth in
Scenario B while preserving MMLU-Pro accuracy" is a hypothesis. "Try SGLang" is
not a hypothesis. "SGLang's scheduler may improve Scenario C constant/poisson
throughput at the same quality gate and model length" is a hypothesis.

## Hardware-Specific Caution

The kernel competition was heavily shaped by Blackwell-era GPU behavior,
Triton code generation, FP8 support, tensor-core paths, and cache behavior.
Some lessons are portable principles; some are not. SENPAI should separate:

- Measured facts on the current pod and GPU.
- Hardware-specific observations from RTX PRO 6000 shakedowns.
- Leaderboard-comparable H100 observations.
- General serving-system principles.

Do not import a kernel-level claim into InferenceBench unless it has been
validated in the actual serving stack. The useful habit is the method: inspect,
hypothesize, measure, log, confirm.

## Minimal Durable Artifacts For A Strong SENPAI Run

A two-hour InferenceBench run should leave enough state that the next run can
be smarter:

- `BASELINE.md`: live best per scenario, with exact launcher and W&B IDs.
- PR comments: assignment, partial steering, and terminal `SENPAI-RESULT`.
- W&B runs: raw metrics, hardware, scenario, launcher slug, and group.
- `senpai/research/<topic>.md`: compact lessons that survived measurement.
- Launcher files: every candidate worth preserving under `senpai/launchers/`.
- Conversation logs: archived advisor/student `.claude` logs for diagnosis.

The auto-gpu-kernel winner succeeded because each iteration made the next
iteration better informed. That is the standard SENPAI should hold itself to.
