# Qwen3.6-35B-A3B — EXL3 quants on ExLlamaV3 (Radeon 9070 XT)

Status: **WIP** (runs dated 2026-11-12). Testing two EXL3 quantizations, 5.00 bpw
and 6.00 bpw, each with and without speculation. The 6.00 bpw quant has been
benchmarked three ways so far: no spec, MTP (draft from the model's own MTP
layers), and a separate DFlash2 draft. The 5.00 bpw pairs are still pending.

## Setup

- **Model:** Qwen3.6-35B-A3B (~35B MoE, A3B active).
- **Quantizations:** `q5-exl3-5.00bpw`, `q6-exl3-6.00bpw-hb6` (local, `../models`).
- **DFlash2 draft:** `q-dflash2-35b-exl3-3.00bpw` (local).
- **Engine:** ExLlamaV3 (same fork/build as the Ornith 20:53Z run — see
  [README.md](README.md)).
- **Bench tool / protocol:** same llm-bench protocol as the README (prefill at
  fixed lengths, decode 512 tokens, 5 runs, medians). So far only the 4k and
  16k prefill lengths were run, plus short-context decode (~93-prompt).

### Per-run config (6.00 bpw)

| | no speculation | MTP | DFlash2 draft |
| --- | --- | --- | --- |
| Run time | 2026-11-12 09:10:45Z | 09:15:48Z | 09:39:39Z |
| `cpu_moe_split_experts` | 145 | 150 | 155 |
| Draft | — | `draft_mode: mtp`, Q4 draft cache, 4 draft tokens | `draft_mode: model`, `q-dflash2-35b-exl3-3.00bpw`, Q4 draft cache, 5 draft tokens |
| `dynamic_draft` / `draft_confidence` | — | on / 0.4 | on / 0.4 |
| VRAM | ~15.4 GB | ~15.4 GB | ~15.2 GB |

Common to all runs: `backend: exllamav3`, `gpu_split_auto` (no GPU split),
`autosplit_reserve: 768` MB, KV cache Q8 @ 131 072 tokens, `max_seq_len: 130 816`,
`max_batch_size: 1`. Sampling overrides: T 0.6, top-p 0.95, top-k 20, min-p 0,
penalties 0.

### Draft configs (as run 2026-11-12)

MTP (draft comes from the main quant's MTP layers; no separate draft model):

```yaml
draft_model:
  draft_mode: mtp
  draft_cache_mode: Q4
  draft_num_tokens: 4
  dynamic_draft: true
  draft_confidence: 0.4
```

DFlash2 (separate 3.0 bpw draft):

```yaml
draft_model:
  draft_mode: model
  draft_model_dir: ../models
  draft_model_name: q-dflash-2-35b-exl3-3.00bpw
  draft_cache_mode: Q4
  draft_num_tokens: 5
  dynamic_draft: true
  draft_confidence: 0.4
```

(no-spec run: no `draft_model` block.)

## Results — 6.00 bpw (medians of 5 runs)

### Prefill

| Prompt length | no spec | MTP | DFlash2 |
| ---: | ---: | ---: | ---: |
| 4 000 | **996 tok/s** (TTFT 4.16 s) | 907 tok/s (TTFT 4.57 s) | 921 tok/s (TTFT 4.50 s) |
| 16 000 | **1 003 tok/s** (TTFT 16.24 s) | 924 tok/s (TTFT 17.63 s) | 922 tok/s (TTFT 17.68 s) |

### Decode, tok/s

| Context length | no spec | MTP | DFlash2 |
| ---: | ---: | ---: | ---: |
| ~90 (short) | 68.8 (67.1–71.0) | 94.0 (87.6–97.0) | **97.1** (94.6–99.4) |
| 4 000 | 68.9 (68.5–70.0) | 86.3 (78.0–91.1) | **99.0** (93.5–107.1) |
| 16 000 | 67.2 (65.9–67.6) | 82.4 (75.7–84.2) | **88.5** (85.1–98.0) |

### Pending

| | q5-exl3-5.00bpw no spec | q5-exl3-5.00bpw MTP | q5-exl3-5.00bpw DFlash2 | q6 no spec | q6 MTP | q6 DFlash2 |
| --- | --- | --- | --- | --- | --- | --- |
| Runs | — | — | — | 09:10:45Z | 09:15:48Z | 09:39:39Z |
| Results | TODO | TODO | TODO | see above | see above | see above |

## Findings so far

1. **Speculation is a big decode win on this model/engine** — unlike the
   Ornith llama.cpp result where MTP slowed decode down. Short-context decode
   jumps 68.8 → 94.0 (MTP, +37%) / 97.1 (DFlash2, +41%).
2. **DFlash2 beats MTP on decode at every context tested** — 97.1 vs 94.0
   short, 99.0 vs 86.3 at 4k (+15%), 88.5 vs 82.4 at 16k (+7%). The gap widens
   at 4k, consistent with more hunks per step being accepted from the
   separate 3.0 bpw draft (5 draft tokens vs 4).
3. **Speculation costs prefill bandwidth** — MTP ~9% slower (907/924 vs
   996/1 003 at 4k/16k); DFlash2 is similar (921/922) with a ~4.5 s TTFT at 4k
   vs 4.16 s no-spec.
4. **VRAM is essentially unchanged across modes** — ~15.4 GB no-spec and MTP,
   ~15.2 GB with the DFlash2 draft.
5. Decode spread is slightly wider with drafts (short: 67.1–71.0 no-spec vs
   87.6–97.0 MTP / 94.6–99.4 DFlash2).

## Caveats

- Only 6.00 bpw data exists in `results/` so far; 5.00 bpw is expected to land
  next (planned to compare quant impact at the same offload/draft settings).
- Only 4k/16k prefill lengths and short/4k/16k decode were run; 32k–128k still
  outstanding.
- `prompt_tokens unchecked (non-Qwen model)` flag applies to all runs.
- Medians over 5 runs each; no early-stop runs were seen.
- `cpu_moe_split_experts` differs per run (145 / 150 / 155) — the draft variant
  loads extra weights, so absolute numbers across spec modes conflate offload
  level and speculation.

## Raw results

- `exllamav3-20261112T091045Z-q6-exl3-6-00bpw-hb6.json` (no speculation)
- `exllamav3-20261112T091548Z-q6-exl3-6-00bpw-hb6.json` (MTP)
- `exllamav3-20261112T093939Z-q6-exl3-6-00bpw-hb6.json` (DFlash2 draft)

(Also in `../llm-bench/results/` under their original
`20261112T…-q6-exl3-6-00bpw-hb6.json` names.)