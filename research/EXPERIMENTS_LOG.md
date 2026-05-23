# SENPAI Research Results — ib-20260523-rerun-r5

This log records every reviewed PR for the rerun-r5 launch, with the
hypothesis, results, and post-review commentary. Add new entries at the top.

## Round 1 (2026-05-23)

Three concurrent PRs target Scenario C. Results will be filled in as the
advisor reviews each.

- PR #24 — r5-frieren — Bootstrap Scenario C torch baselines + MMLU-Pro quality.
  Status: in flight (status:wip). Tooling PR; on-disk baseline artifacts are
  the deliverable.
- PR #25 — r5-fern — vLLM Scenario C throughput launcher v1 (FP16 KV cache,
  max-num-seqs=256). Status: in flight (status:wip). Gated on PR #24
  SLOT-FREE.
- PR #27 — r5-tanjiro — vLLM Scenario C launcher with FP8 KV cache +
  max-num-seqs=512. Status: in flight (status:wip). Gated on PR #25
  SLOT-FREE.
