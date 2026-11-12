# Qwen3.6-35B-A3B — EXL3 quants on ExLlamaV3 (Radeon 9070 XT)

Status: **WIP** (runs dated 2026-11-12). Testing two EXL3 quantizations, 5.00 bpw
and 6.00 bpw, each with and without MTP speculation. Only the **6.00 bpw** runs
have landed in results so far; the 5.00 bpw pairs are pending.

## Setup

- **Model:** Qwen3.6-35B-A3B (~35B MoE, A3B active).
- **Quantizations:** `q5-exl3-5.00bpw`, `q6-exl3-6.00bpw-hb6` (local, `../models`).
- **Engine:** ExLlamaV3 (same fork/build as the Ornith 20:53Z run — see
  [README.md](README.md)).
- **Bench tool / protocol:** same llm-bench protocol as the README (prefill at
  fixed lengths, decode 512 tokens, 5 runs, medians). So far only the 4k and
  16k prefill lengths were run, plus short-context decode (~93-prompt).

### Per-run config (6.00 bpw)

| | no MTP | MTP |
| --- | --- | --- |
| `cpu_moe_split_experts` | 145 | 150 |
| MTP draft | — | `draft_mode: mtp`, Q4 draft cache, 4 draft tokens, `dynamic_draft: true`, `draft_confidence: 0.4` |
| VRAM | ~15.4 GB | ~15.4 GB |

Common to all runs: `backend: exllamav3`, `gpu_split_auto` (no GPU split),
`autosplit_reserve: 768` MB, KV cache Q8 @ 131 072 tokens, `max_seq_len: 130 816`,
`max_batch_size: 1`. Sampling overrides: T 0.6, top-p 0.95, top-k 20, min-p 0,
penalties 0.

Full MTP config (as run 2026-11-12):

```yaml
model:
  model_dir: ../models
  model_name: q6-exl3-6.00bpw-hb6
  backend: exllamav3
  max_seq_len: 130816
  cache_size: 131072
  cache_mode: Q8
  tensor_parallel: false
  gpu_split_auto: true
  autosplit_reserve: [768]
  cpu_moe_split_experts: 150
  max_batch_size: 1

draft_model:
  draft_mode: mtp
  # draft_model_dir: ../models
  # draft_model_name: Ornith-1.5-35B-A3B-DFlash2-EXL3-3.0bpw
  draft_cache_mode: Q4
  draft_num_tokens: 4
  dynamic_draft: true
  draft_confidence: 0.4
```

(The no-MTP run is identical except for the `draft_model` block and
`cpu_moe_split_experts: 145`.)

## Results — 6.00 bpw (medians of 5 runs)

### Prefill

| Prompt length | no MTP | MTP |
| ---: | ---: | ---: |
| 4 000 | **996 tok/s** (TTFT 4.16 s) | 907 tok/s (TTFT 4.57 s) |
| 16 000 | **1 003 tok/s** (TTFT 16.24 s) | 924 tok/s (TTFT 17.63 s) |

### Decode, tok/s

| Context length | no MTP | MTP |
| ---: | ---: | ---: |
| ~90 (short) | 68.8 (67.1–71.0) | **94.0** (87.6–97.0) |
| 4 000 | 68.9 (68.5–70.0) | **86.3** (78.0–91.1) |
| 16 000 | 67.2 (65.9–67.6) | **82.4** (75.7–84.2) |

### Pending

| | q5-exl3-5.00bpw no MTP | q5-exl3-5.00bpw MTP | q6-exl3-6.00bpw-hb6 no MTP | q6-exl3-6.00bpw-hb6 MTP |
| --- | --- | --- | --- | --- |
| Runs | — | — | 2026-11-12 09:10:45Z | 2026-11-12 09:15:48Z |
| Results | TODO | TODO | see above | see above |

## Findings so far

1. **MTP helps decode a lot on this model/engine** — unlike the Ornith llama.cpp
   result where MTP slowed decode down. Short-context decode jumps 68.8 →
   94.0 tok/s (+37%); 4k 68.9 → 86.3 (+25%); 16k 67.2 → 82.4 (+23%).
2. **MTP costs prefill bandwidth** — ~9% slower with the draft (907/924 vs
   996/1 003 tok/s at 4k/16k), so TTFT grows ~10% (4.57 vs 4.16 s at 4k).
3. **VRAM is unchanged by MTP** — ~15.4 GB for both 6.00 bpw runs.
4. Decode spread is slightly wider with MTP (87.6–97.0) than without
   (67.1–71.0) at short context.

## Caveats

- Only 6.00 bpw data exists in `results/` so far; 5.00 bpw is expected to
  land next (planned to compare quant impact at the same offload/draft settings).
- Only 2 000/4 000-prompt prefill lengths and short/4k/16k decode were run;
  32k–128k still outstanding.
- `prompt_tokens unchecked (non-Qwen model)` flag applies to both runs.
- Medians over 5 runs each; no early-stop runs were seen.
- While draft paths for a DFlash2 draft were present in the models dir
  (`q-dflash2-35b-exl3-3.00bpw`), the config used as documented above has them
  commented out — MTP drafts come from the main quant's own MTP layers.

## Raw results

- `exllamav3-20261112T091045Z-q6-exl3-6-00bpw-hb6.json` (no MTP)
- `exllamav3-20261112T091548Z-q6-exl3-6-00bpw-hb6.json` (MTP)

(Also in `../llm-bench/results/` under their original
`20261112T…-q6-exl3-6-00bpw-hb6.json` names.)