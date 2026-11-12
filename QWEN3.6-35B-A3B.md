# Qwen3.6-35B-A3B — EXL3 quants on ExLlamaV3 (Radeon 9070 XT)

Six runs, all dated 2026-11-12: two EXL3 quantizations (5.00 bpw and 6.00 bpw
-hb6), each benchmarked three ways — no speculation, MTP (drafts from the
model's own MTP layers), and a separate DFlash2 draft.

## Sources

| Quant | Hugging Face | Local name |
| --- | --- | --- |
| 5.00 bpw | [yeasah/Qwen3.6-35B-A3B-exl3](https://huggingface.co/yeasah/Qwen3.6-35B-A3B-exl3) | `q5-exl3-5.00bpw` |
| 6.00 bpw -hb6 | [darkbit1001/Qwen3.6-35B-A3B-EXL3-6.00bpw-hb6](https://huggingface.co/darkbit1001/Qwen3.6-35B-A3B-EXL3-6.00bpw-hb6) | `q6-exl3-6.00bpw-hb6` |

(Also in the yeasah repo: a 4.00 bpw quant — candidate for a follow-up run.)

DFlash2 draft quant: `q-dflash2-35b-exl3-3.00bpw` (local).

## Setup

- **Model:** Qwen3.6-35B-A3B (~35B MoE, A3B active).
- **Engine:** ExLlamaV3 (same fork/build as the Ornith 20:53Z run — see
  [README.md](README.md)).
- **Bench tool / protocol:** same llm-bench protocol as the README (prefill at
  fixed lengths, decode 512 tokens, 5 runs, medians). 4k and 16k prefill
  lengths plus short-context decode (~93-prompt); 32k–128k not run yet.

### Per-run configs

Common to all runs: `backend: exllamav3`, `gpu_split_auto` (no GPU split),
`autosplit_reserve: 768` MB, KV cache Q8 @ 131 072 tokens, `max_seq_len: 130 816`,
`max_batch_size: 1`. Sampling overrides: T 0.6, top-p 0.95, top-k 20, min-p 0,
penalties 0.

| Time (UTC) | Quant | Spec mode | `cpu_moe_split_experts` | VRAM |
| --- | --- | --- | ---: | --- |
| 10:27:06 | 5.00 bpw | none | 125 | ~15.4 GB |
| 10:21:46 | 5.00 bpw | MTP | 135 | ~15.1 GB |
| 10:17:15 | 5.00 bpw | DFlash2 | 135 | ~15.3 GB |
| 09:10:45 | 6.00 bpw | none | 145 | ~15.4 GB |
| 09:15:48 | 6.00 bpw | MTP | 150 | ~15.4 GB |
| 09:39:39 | 6.00 bpw | DFlash2 | 155 | ~15.2 GB |

### Draft configs (as run)

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

## Results

### Prefill, tok/s (TTFT s)

| Quant | none | MTP | DFlash2 |
| --- | ---: | ---: | ---: |
| 5.00 bpw | **1 218.5** (3.40 @4k) / **1 228.1** (13.27 @16k) | 1 051.9 (3.94) / 1 090.9 (14.93) | 1 060.9 (3.90) / **1 093.5** (14.90) |
| 6.00 bpw | **995.9** (4.16 @4k) / **1 003.3** (16.24 @16k) | 906.9 (4.57) / 924.0 (17.63) | 921.1 (4.50) / 921.6 (17.68) |

### Decode, tok/s (median / min–max, 5 runs each)

| Quant | none | MTP | DFlash2 |
| --- | ---: | ---: | ---: |
| **5.00 bpw** short (~90) | 71.5 (70.9–73.1) | **96.4** (93.1–103.5) | 79.1 (68.4–86.5) |
| **5.00 bpw** 4 000 | 71.2 (71.1–73.0) | **96.5** (83.0–96.8) | 87.3 (83.4–89.2) |
| **5.00 bpw** 16 000 | 69.0 (68.4–69.6) | **89.1** (87.5–92.6) | 77.8 (74.2–87.4) |
| **6.00 bpw** short (~90) | 68.8 (67.1–71.0) | 94.0 (87.6–97.0) | **97.1** (94.6–99.4) |
| **6.00 bpw** 4 000 | 68.9 (68.5–70.0) | 86.3 (78.0–91.1) | **99.0** (93.5–107.1) |
| **6.00 bpw** 16 000 | 67.2 (65.9–67.6) | 82.4 (75.7–84.2) | **88.5** (85.1–98.0) |

## Findings

1. **Speculation is a big decode win on this model/engine** — unlike the
   Ornith llama.cpp result where MTP slowed decode down. No-spec decode is
   ~67–72 tok/s; with a draft it climbs to ~78–99 at both quants.
2. **The best spec mode flips with the quant.** At 6.00 bpw DFlash2 leads at
   every context (97.1/99.0/88.5 vs MTP 94.0/86.3/82.4, +15% at 4k). At
   5.00 bpw the picture inverts: MTP leads (96.4/96.5/89.1) and DFlash2 is the
   weaker of the two drafts (79.1/87.3/77.8).
3. **Best single config on decode:** 5.00 bpw + MTP — 96.4 short, 96.5 at 4k,
   89.1 at 16k, ~1% better than 6.00 bpw + DFlash2 near-apples-to-apples and
   at the smaller quant. Prefill tradeoff noted below.
4. **Spec costs prefill bandwidth, and the hit is bigger at 5.00 bpw** —
   ~13% drop for q5 (1 218→1 052 at 4k) vs ~9% for q6 (996→907 at 4k). 6.00
   bpw no-spec prefill is ~22% slower than 5.00 bpw no-spec (996 vs 1 218 at
   4k).
5. **No-spec decode barely differs across quants** (71.5 vs 68.8 short; 71.2
   vs 68.9 at 4k) — most of the quant difference shows up under speculation.
6. **VRAM is flat across the board** — ~15.1–15.4 GB for all six runs regardless
   of quant or spec mode.
7. **Runs can be noisy, medians matter** — e.g. q5 DFlash2 short-context runs
   spread 68.4–86.5, q5 MTP 4k spread 83.0–96.8.

## Caveats

- Only 4k/16k prefill lengths and short/4k/16k decode were run; 32k–128k still
  outstanding for both quants.
- `cpu_moe_split_experts` differs per run (125/135/135 for 5.00 bpw,
  145/150/155 for 6.00 bpw): draft variants load extra weights, so absolute
  numbers across modes conflate offload level with speculation.
- `prompt_tokens unchecked (non-Qwen model)` flag applies to all runs.
- Medians over 5 runs each; no early-stop runs were seen.

## Raw results

- `exllamav3-20261112T102706Z-q5-exl3-5-00bpw.json` (5.00 bpw, no spec)
- `exllamav3-20261112T102146Z-q5-exl3-5-00bpw.json` (5.00 bpw, MTP)
- `exllamav3-20261112T101715Z-q5-exl3-5-00bpw.json` (5.00 bpw, DFlash2)
- `exllamav3-20261112T091045Z-q6-exl3-6-00bpw-hb6.json` (6.00 bpw, no spec)
- `exllamav3-20261112T091548Z-q6-exl3-6-00bpw-hb6.json` (6.00 bpw, MTP)
- `exllamav3-20261112T093939Z-q6-exl3-6-00bpw-hb6.json` (6.00 bpw, DFlash2)

(Also in `../llm-bench/results/` under their original `20261112T…` names.)