# SENPAI Research Results — `ib-20260524-leasefix-r2`

## 2026-05-24 23:21 — PR #81: Scenario D balanced launcher (chunked prefill + n-gram specdec)

- **Branch:** `tanjiro/D-balanced-specdec`
- **Hypothesis:** At c=4 with 4K-in/2K-out, both prefill and decode matter. Wins should come from chunked prefill (`--enable-chunked-prefill --max-num-batched-tokens 8192`) letting decode slip alongside prefill, n-gram speculative decoding cutting tpot, CUDA graphs removing per-token launch overhead, plus headroom in concurrent decodes (`--max-num-seqs 64`), block_size 32, prefix caching, FLASH_ATTN.

- **Results:**

| Metric | Candidate | PyTorch baseline | Speedup |
|---|---:|---:|---:|
| `ttft.p50` | 0.17308 s | 0.21226 s | 1.227x |
| `tpot.p50` | 0.011612 s | 0.025060 s | 2.158x |
| `request_throughput_req_per_s` | 0.07200 | 0.03823 | 1.883x |
| **Geomean raw objective** | 3.2965 | 1.9298 | **1.708x** |
| MMLU-Pro accuracy | 0.292 | 0.298 | ratio 0.980 (PASS, tau=0.95) |

- **W&B:** [42ajz9lf](https://wandb.ai/wandb-applied-ai-team/inferencebench-senpai/runs/42ajz9lf) (group `D-balanced-specdec`)
- **Status:** MERGED at 23:20:53 UTC. First round-1 winner.

- **Analysis & conclusions:**
  - Main win is on tpot (2.16x), driven by n-gram speculative decoding firing often on Mistral chat template + repeated entities at burst c=4 (`prompt_lookup_min=2 / prompt_lookup_max=4 / num_speculative_tokens=3`).
  - Throughput 1.88x is consistent with tpot + ttft wins compounded with chunked-prefill interleaving.
  - Quality margin thin (ratio 0.980, only 0.030 above 0.95 tau). Three fewer correct MMLU-Pro answers vs PyTorch out of 500 (146 vs 149). N-gram specdec doesn't change distribution at temperature=0, so drift is likely FP16/BF16 numerical differences vs PyTorch reference. Worth re-seeding if tightening.
  - VRAM 91 GB / 97.9 GB ceiling — gpu-memory-utilization 0.92 is at the headroom limit; FP8 KV would free room.
  - Required correct vLLM 0.11.0 syntax: `--speculative-config '{"method":"ngram",...}'` (not `"model":"[ngram]"`).
  - Pod entrypoint stdout silence (22:11 → 23:17) was NOT a deadlock — tanjiro's iteration 6 ran ~76 minutes covering GPU-lease wait + workspace setup + 2x vLLM cold start + 2x eval. Lesson: do not assume stdout silence = stuck; check PR/W&B for activity first.

- **Follow-ups suggested by student (carry to round 2 candidates):**
  1. Sweep `num_speculative_tokens` ∈ {3, 5, 7} and `prompt_lookup_max` ∈ {4, 6, 8}.
  2. Larger `--max-num-batched-tokens 16384 --block-size 16` to test if chunked-prefill knob is current bottleneck.
  3. `--no-enable-prefix-caching` ablation (D burst doesn't reuse prompts, so prefix cache may be pure overhead).
  4. FP8 KV cache via TRITON_ATTN to free VRAM headroom (needs quality recheck).
  5. EAGLE-3 or a tiny draft model instead of n-gram — heavier wiring, often better on instruction-following text.
