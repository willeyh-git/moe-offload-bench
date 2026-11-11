# moe-offload-bench

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

- **Model:** Ornith 1.5 35B-A3B (~35B params, MoE), different quantizations per engine.
- **Engines:** llama.cpp (APEX GGUF), exllamav3 (EXL3), Polystrata (APEX GGUF).
- **Bench protocol:** prefill at fixed prompt lengths, then decode (512 tokens)
  at short context and at each prefill length (prompt already cached). 5 runs per
  size where possible; medians reported. One shared corpus (sha256
  `58a7756f…`). Runs dated 2026-11-11.
- **Hot expert cache mechanism differs per engine** — see configs below.

## Config summary

| Engine | Model file | GPU offload / expert offload | KV cache | Context | Sampling |
| --- | --- | --- | --- | --- | --- |
| llama.cpp (no MTP) | Ornith-1.5-35B-A3B-APEX GGUF | `-fa on`, `--n-cpu-moe 25`, `--moe-cache-mib 4096` (hot expert cache) | q8_0 k/v | 131 072 | T 0.6, top-p 0.95, top-k 20, min-p 0 |
| llama.cpp (MTP) | same + MTP drafts | `-fa on`, `--n-cpu-moe 30`, `--moe-cache-mib 4096`, `--spec-type draft-mtp,ngram-mod`, draft q4_0 k/v, up to 3 draft tokens | q8_0 k/v | 131 072 | same |
| exllamav3 | Ornith-1.5-35B-A3B-EXL3 4.00 bpw (24 GB) | `gpu_split_auto`, `cpu_moe_split_experts: 100` (100 experts resident on CPU) | Q8, 130 816 tokens, 768 MB autosplit reserve | 130 816 | same; spec draft: DFlash2 EXL3 3.0 bpw, 4 draft tokens, dynamic |
| Polystrata | Ornith-1.5-35B-A3B-APEX-I-QUALITY GGUF + `expert-profile.txt` | expert profile + hot cache (`STRATA_RESIDENT_PIN=0`, `STRATA_ROUTE_TAIL_SKIP=0`) | `--kv-whole` | 131 072 | same |

Full configs: [`LLAMA_CPP_CONF.md`](LLAMA_CPP_CONF.md),
[`EXLLAMAV3_CONF.md`](EXLLAMAV3_CONF.md), [`POLYSTRATA_CONF.md`](POLYSTRATA_CONF.md).

### Engine binaries / sources

| Engine | Source | Commit used |
| --- | --- | --- |
| llama.cpp | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (ROCm build, `BUILD-ROCM.sh`) | `f7b886fe` (2026-11-11) |
| Polystrata | [simplexai-labs/PolyStrata](https://github.com/simplexai-labs/PolyStrata) | `e4ea22a3` (2026-11-10) |
| exllamav3 | fork [willeyh-git/exl3-rdna4-recipe](https://github.com/willeyh-git/exl3-rdna4-recipe) (forked from [parth-thakre/exl3-rdna4-recipe](https://github.com/parth-thakre/exl3-rdna4-recipe)) | `fe77c7b4` (2026-11-02, fork base, ExLlamaV3 v1.6.6) |

## Results

### Prefill (tok/s, TTFT s)

| Prompt length | llama.cpp no-MTP | llama.cpp MTP | exllamav3 | Polystrata |
| ---: | ---: | ---: | ---: | ---: |
| 4 000 | 966 (4.29) | 824 (5.02) | 1 309 (3.16) | **1 770 (2.34)** |
| 16 000 | 1 032 (15.78) | 881 (18.49) | 1 395 (11.68) | **1 963 (8.30)** |
| 32 000 | — | — | 1 388 (23.56) | **1 614 (20.26)** |
| 64 000 | — | — | 1 321 (49.48) | **1 088 (60.08)** |
| 96 000 | — | — | 1 225 (80.45) | 832 (118.35) |
| 126 000 | — | — | — (rejected: 128k prompt over 130 816 ctx) | 710 (181.78) |

### Decode, tok/s (median over runs)

| Context length | llama.cpp no-MTP | llama.cpp MTP | exllamav3 | Polystrata |
| ---: | ---: | ---: | ---: | ---: |
| ~16 (short) | 62.7 | 52.5 | 34.6 | **92.7** |
| 4 000 | 61.7 | 58.1 | 36.9 | **93.1** |
| 16 000 | 57.0 | 50.8 | 35.0 | **84.5** |
| 32 000 | — | — | 33.5 | **74.6** |
| 64 000 | — | — | 32.3 | **58.5** |
| 96 000 | — | — | 31.7 | **49.8** |
| 126 000 | — | — | — | **43.2** |

"-" = not run for that config (llama.cpp runs covered only 4k/16k; exllamav3
rejected the 128k prompt).

## Findings so far

1. **Polystrata is the fastest engine on every metric measured** — ~1.5× llama.cpp
   and ~2.7× exllamav3 in short-context decode (92.7 vs 62.7 vs 34.6 tok/s), and
   the strongest prefill at every context length (up to ~1.9× llama.cpp, ~1.4×
   exllamav3 at short context). Its expert-profile-driven hot cache appears to pay
   off hard on RDNA4.
2. **Decode degrades with context — but slopes differ.** Polystrata falls from
   ~93 to ~43 tok/s at 126k context (~54% drop). exllamav3's decode is much weaker
   in absolute terms but far flatter (~37 → ~32 tok/s, drop only just above 16k).
   llama.cpp sits in between at the lengths measured (62.7 → 57).
3. **Prefill throughput vs. context:** exllamav3 stays remarkably steady at
   ~1.2–1.4k tok/s out to 96k; Polystrata degrades from 1 963 (16k) to 710 (126k),
   the largest long-context slowdown across all three engines (KV-heavy at
   `--kv-whole`). llama.cpp is the slowest measured prefill (824–1 032 tok/s) and
   looks roughly flat between the two lengths it covers (966 at 4k, 1 032 at 16k).
4. **MTP makes llama.cpp slower, not faster, in this stack.** With MTP the no-MTP
   baseline is beaten on prefill (824/881 vs 966/1 032 tok/s) and on decode
   (52.5 vs 62.7 tok/s short context; 50.8 vs 57.0 at 16k) — the CPU-side
   speculation cost outweighs draft acceptance at 30 CPU-resident experts.
5. **Hot cache benefit is real where it exists.** llama.cpp's 4 096 MiB
   `--moe-cache-mib` keeps short-context decode within ~2 tok/s of the short-prompt
   numbers at 4k/16k; exllamav3 shows a similar near-flat profile (36.9 at 4k vs
   34.6 short). Polystrata's profile + cache is what separates it entirely.
6. **exllamav3 is the outlier in decode** (34.6 tok/s short context, ~1.8× below
   llama.cpp despite a stronger per-token pipeline for prefill) — even with its
   DFlash2 draft model configured. Worth verifying whether spec decode was
   actually active during the benchmarked runs.

## Caveats

- Medians over ≤5 runs; TTFT std-dev across samples was low in most cases.
- A few runs ended early (`finish_reason: stop`) and are flagged in the JSON files;
  medians include them.
- Token-count checks (`prompt_tokens unchecked`) apply to all engines; benchmark
  prompts are synthetic corpus text.
- Each engine uses a different quantization (APEX GGUF vs EXL3 4.00 bpw), so
  cross-engine comparisons mix engine + format effects.

## Raw results

- `exllamav3-ornith-1-5-35b-a3b-exl3-4-00bpw-24gb.json`
- `llamacpp-ornith-1-5-35b-a3b-apex-i-quality-no-mtp.json`
- `llamacpp-ornith-1-5-35b-a3b-apex-i-quality-mtp.json`
- `polystrata-ornith-1-5-35b-a3b-apex-i-quality.json`