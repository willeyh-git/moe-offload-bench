# Exploring MoE CPU offloading and hot cache experts

Exploring MoE expert CPU offloading and hot expert caches across different
inference engines on RDNA4 (Radeon 9070 XT).

## System

| Component | Spec |
| --- | --- |
| OS | `7.3.1-1-cachyos #1 SMP PREEMPT_DYNAMIC Sat, 31 Oct 2026 15:33:05 +0000 x86_64 GNU/Linux` |
| CPU | AMD Ryzen 7 5700X3D (8c/16t) |
| RAM | 32 GB DDR4-3200 (4 × 8 GB) |
| GPU | Gigabyte Radeon 9070 XT Gaming OC (power limit 300 W) |

## Scope

Per-model results live in their own files:

- **Ornith 1.5 35B-A3B** (~35B params, MoE) across llama.cpp (APEX GGUF),
  exllamav3 (EXL3) and Polystrata (APEX GGUF), different quantizations per
  engine — configs, results, findings and raw runs in
  [ORNITH1.5-35B-A3B.md](ORNITH1.5-35B-A3B.md).
- **Qwen3.8-Flash-Next "Coder"** (IQ1_M, Polystrata, nov-7 build, `--spec 2`),
  single run — see [QWEN3.8-FLASH_NEXT.md](QWEN3.8-FLASH_NEXT.md).
- **Qwen3.6-35B-A3B** EXL3 quants and GGUF runs — see
  [QWEN3.6-35B-A3B.md](QWEN3.6-35B-A3B.md).

### Bench protocol

- **Bench tool:** [willeyh-git/llm-bench](https://github.com/willeyh-git/llm-bench) (Go, commit `f5de163`).
  It drives each engine's OpenAI-compatible server through the protocol below;
  JSON results land in `results/`.
- **Bench protocol:** prefill at fixed prompt lengths (4k, 16k, 32k, 64k, 96k, 128k
  targets), then decode (512 tokens) at short context and at each prefill length
  (prompt already cached). 5 runs per size where possible; medians reported. One
  shared corpus (sha256 `58a7756f…`).
- **Hot expert cache mechanism differs per engine** — see the per-model files
  and the engine config docs.

Engine config docs: [`LLAMA_CPP_CONF.md`](LLAMA_CPP_CONF.md),
[`EXLLAMAV3_CONF.md`](EXLLAMAV3_CONF.md), [`POLYSTRATA_CONF.md`](POLYSTRATA_CONF.md).

## Raw results (master list)

The per-model files list and annotate their own JSON files; this is the flat
list, newest first.

- `20261112T155239Z-qwen3-6-35b-a3b-ud-q4-k-xl.json` (Qwen3.6 UD Q4_K_XL, PolyStrata — [QWEN3.6-35B-A3B.md](QWEN3.6-35B-A3B.md))
- `20261112T151400Z-qwen3-6-35b-a3b-ud-q4-k-xl.json` (Qwen3.6 UD Q4_K_XL, Strata Unsloth build)
- `20261112T134157Z-qwen-3-6-q4-llama.json` (Qwen3.6, llama.cpp no-drafter)
- `20261112T132841Z-qwen-3-6-q4-llama.json` (Qwen3.6, llama.cpp MTP)
- `20261112T171930Z-qwen3-8-flash-next-coder-iq1-m.json` (Qwen3.8-Flash-Next "Coder" IQ1_M, Polystrata nov-7 build, `--spec 2` — [QWEN3.8-FLASH_NEXT.md](QWEN3.8-FLASH_NEXT.md))
- `exllamav3-20261112T124935Z-q5-exl3-5-00bpw.json`
- `exllamav3-20261112T121921Z-q5-exl3-5-00bpw.json`
- `exllamav3-20261112T114437Z-q4-exl3-4-00bpw.json`
- `exllamav3-20261112T112000Z-q4-exl3-4-00bpw.json`
- `exllamav3-20261112T110224Z-q4-exl3-4-00bpw.json`
- `exllamav3-20261112T105611Z-q4-exl3-4-00bpw.json`
- `exllamav3-20261112T105007Z-q4-exl3-4-00bpw.json`
- `exllamav3-20261112T102706Z-q5-exl3-5-00bpw.json`
- `exllamav3-20261112T102146Z-q5-exl3-5-00bpw.json`
- `exllamav3-20261112T101715Z-q5-exl3-5-00bpw.json`
- `exllamav3-20261112T093939Z-q6-exl3-6-00bpw-hb6.json`
- `exllamav3-20261112T091548Z-q6-exl3-6-00bpw-hb6.json`
- `exllamav3-20261112T091045Z-q6-exl3-6-00bpw-hb6.json`
- Ornith 1.5 35B-A3B: nine JSON files, annotated in
  [ORNITH1.5-35B-A3B.md](ORNITH1.5-35B-A3B.md)