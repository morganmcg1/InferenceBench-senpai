# SENPAI Baseline Ledger — `ib-20260525-three1-r1`

Live advisor-owned baseline for this launch. Update only from clean
full-evaluation runs tied to this advisor branch and PR with the quality gate
passing.

- Target repo: `morganmcg1/InferenceBench-senpai`
- Advisor branch: `ib-20260525-three1-r1`
- Target base branch: `codex/inferencebench-senpai-target`
- Research tag: `ib-20260525-three1-r1`
- Base model: `mistralai/Mistral-7B-Instruct-v0.3`
- Time budget per launch: 2 hours
- Hardware: 1 GPU shared by 3 students (`frieren`, `fern`, `tanjiro`).
  Current shakedown GPU class: **RTX PRO 6000 Blackwell ~96GB VRAM**.
  RTX PRO 6000 results are shakedown evidence only — **not** leaderboard-comparable to the public H100 80GB setting.
- W&B project: `wandb-applied-ai-team/inferencebench-senpai`
- Scoring assets imported from `/mnt/new-pvc/inferencebench-senpai/scoring-assets/rtxpro6000-seed248`.

## PyTorch baseline raw metrics (seed 248, RTX PRO 6000)

These are the denominators for `speedup_over_pytorch` on this hardware.

| Scenario | Profile  | ttft.p50 (s) | tpot.p50 (s) | req/s   | gen tok/s | succ/fail |
| -------- | -------- | -----------: | -----------: | ------: | --------: | --------: |
| A        | burst    |       0.4385 |       0.0258 |  0.0709 |     35.65 |     128/0 |
| B        | burst    |       0.0709 |       0.0252 |  0.0133 |     39.19 |      64/0 |
| C        | burst    |       0.0702 |       0.0237 |  0.0845 |     39.19 |     256/0 |
| C        | poisson  |       0.0702 |       0.0240 |  0.0847 |     38.39 |     256/0 |
| C        | constant |       0.0704 |       0.0241 |  0.0849 |     38.62 |     256/0 |
| D        | burst    |       0.2123 |       0.0251 |  0.0382 |     38.21 |      96/0 |

## Current best valid launcher per scenario

| Scenario | Best `speedup_over_pytorch` | Engine | Launcher recipe | W&B run | PR |
| -------- | --------------------------- | ------ | --------------- | ------- | -- |
| A        | **1.240x** (ttft.p50 0.3537s vs PyTorch 0.4385s, 128/128, MMLU-Pro 1.020 PASS) | vLLM | `senpai/launchers/A/fern-tuned-vllm-prefill-noprefix/start_server.sh` — `--max-model-len 12288 --max-num-batched-tokens 16384 --max-num-seqs 16 --gpu-memory-utilization 0.95 --no-enable-prefix-caching --no-enable-chunked-prefill` | `insngzmf` | #105 |
| B        | _none yet_                  | _n/a_  | _n/a_           | _n/a_   | _n/a_ |
| C        | _none yet_                  | _n/a_  | _n/a_           | _n/a_   | _n/a_ |
| D        | _none yet_                  | _n/a_  | _n/a_           | _n/a_   | _n/a_ |

## Public reference snapshot (2026-05-21, H100 80GB, 2h budget — for direction only)

| Method                                | Aggregate | Sc. A | Sc. B | Sc. C | Sc. D |
| ------------------------------------- | --------: | ----: | ----: | ----: | ----: |
| SMAC3 search, 2h vLLM                 |    11.53x | 4.37x | 15.23x | 46.70x | 5.69x |
| TPE search, 2h vLLM                   |    11.25x | 4.48x | 14.76x | 43.46x | 5.58x |
| Random search, 2h vLLM                |    10.20x | 4.21x | 11.34x | 41.81x | 5.42x |
| Best listed agent, Claude Sonnet 4.6  |     8.08x | 3.47x | 12.03x | 33.93x | 3.01x |
| vLLM default, no agent                |     4.05x | 1.25x |  2.25x | 48.69x | 1.96x |
| SGLang default, no agent              |     3.92x | 1.22x |  1.77x | 51.12x | 2.14x |
| HF TGI default, no agent              |     3.30x | 1.14x |  1.37x | 41.94x | 1.80x |
| PyTorch baseline                      |     1.00x | 1.00x |  1.00x |  1.00x | 1.00x |

Headroom on RTX PRO 6000 is expected to track the same direction
(B and A have the largest multiplier room over default vLLM; C is already
close to its tuned ceiling). FP8 weight quantization and FlashInfer must be
proved on this hardware before being used.

## Update history

- 2026-05-25 15:39 UTC — initial BASELINE.md created on launch boot.
- 2026-05-25 17:03 UTC — **PR #105 merged** (fern, Sc. A). First confirmed Sc. A result on RTX PRO 6000: **1.240x** speedup. Launcher: `senpai/launchers/A/fern-tuned-vllm-prefill-noprefix/start_server.sh`. Full eval 128/128, MMLU-Pro n=500 PASS (0.304 vs 0.298 baseline, ratio 1.020). W&B: `insngzmf`. Notes: arm 2 (no prefix caching) marginally beat arm 1 (1.24x vs 1.27x quick); FP8 arm 3 staged on branch as round-2 candidate.
- 2026-05-25 17:31 UTC — **Round 1 complete.** No further baseline updates this launch. PR #106 closed informational (1.246x quick Sc. D, not BASELINE-eligible per full-eval contract). PR #104 closed watchdog kill (2.89x quick ngram on Sc. B banked as round-2 priority — never made it to full eval). PR #107 closed descoped/no-slot. Sc. B/C/D rows remain `_none yet_`. Round-2 pickups in priority order: (1) Sc. B full eval at `senpai/launchers/B/frieren-tuned-vllm-decode-ngram/`, (2) Sc. D full eval at `senpai/launchers/D/tanjiro-tuned-vllm-balanced/`, (3) Sc. A FP8 arm at `senpai/launchers/A/fern-tuned-vllm-prefill-noprefix-fp8/`.

## Update rules

1. Only update "Current best" rows from a full `evaluate.py` run on this
   advisor branch and a PR whose terminal `SENPAI-RESULT` marker passes
   quality, has low failure rate, and reports the matching speedup name.
2. Each update row must reference the exact launcher recipe (path + git SHA),
   the W&B run ID, and the PR number.
3. Record candidate launchers that pass quality but did not beat the current
   best as research signal in PR comments and the experiments log, not in the
   "Current best" table.
