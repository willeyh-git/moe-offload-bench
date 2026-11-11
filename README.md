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
- **Bench tool:** [willeyh-git/llm-bench](https://github.com/willeyh-git/llm-bench) (Go, commit `f5de163`).
  It drives each engine's OpenAI-compatible server through the protocol below;
  JSON results land in `results/`.
- **Bench protocol:** prefill at fixed prompt lengths (4k, 16k, 32k, 64k, 96k, 128k
  targets), then decode (512 tokens) at short context and at each prefill length
  (prompt already cached). 5 runs per size where possible; medians reported. One
  shared corpus (sha256 `58a7756f…`). Runs dated 2026-11-11.
- **Hot expert cache mechanism differs per engine** — see configs below.

### Models

| Engine | Quantization |
| --- | --- |
| llama.cpp, Polystrata | APEX MTP GGUF — [mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF) (quality file: `Ornith-1.5-35B-A3B-APEX-MTP-Quality.gguf`) |
| exllamav3 | [JRamirez-UAB/Ornith-1.5-35B-A3B-EXL3-4.00bpw-24GB](https://huggingface.co/JRamirez-UAB/Ornith-1.5-35B-A3B-EXL3-4.00bpw-24GB); DFlash2 draft requanted to EXL3 (3.0 bpw) from [jzinno/Ornith-1.5-35B-A3B-DFlash2](https://huggingface.co/jzinno/Ornith-1.5-35B-A3B-DFlash2) |

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
| 4 000 | 966 (4.29) | 824 (5.02) | 670 (6.18) | **1 770 (2.34)** |
| 16 000 | 1 032 (15.78) | 881 (18.49) | 1 154 (14.12) | **1 963 (8.30)** |
| 32 000 | — | — | 1 425 (22.94) | **1 614 (20.26)** |
| 64 000 | — | — | **1 344 (48.64)** | 1 088 (60.08) |
| 96 000 | — | — | **1 273 (77.41)** | 832 (118.35) |
| 128 000 | — | — | **1 214 (106.29)** | 710 (181.78) |

### Decode, tok/s (median over runs)

| Context length | llama.cpp no-MTP | llama.cpp MTP | exllamav3 | Polystrata |
| ---: | ---: | ---: | ---: | ---: |
| ~16 (short) | 62.7 | 52.5 | 75.4 | **92.7** |
| 4 000 | 61.7 | 58.1 | 74.9 | **93.1** |
| 16 000 | 57.0 | 50.8 | **85.4** | 84.5 |
| 32 000 | — | — | 67.8 | **74.6** |
| 64 000 | — | — | **61.1** | 58.5 |
| 96 000 | — | — | **52.7** | 49.8 |
| 128 000 | — | — | **50.0** | 43.2 |

"-" = not run. exllamav3 figures are the 15:59Z run; its 32k decode median
includes two early-stop runs (115/208 tokens).

### Throughput vs. context length

![Prefill and decode throughput per context length](plots/benchmarks.png)

Linear axes; short-context decode (prompt ~16 tok) plotted at the left edge.
Generated with `plots/make_plots.py` (`python3` + matplotlib).

## Findings so far

1. **Short context: Polystrata is fastest** — 92.7 tok/s short-context decode
   (vs 75.4 exllamav3, 62.7 llama.cpp no-MTP) and the strongest prefill at short
   prompts (1 770 tok/s at 4k, 1 963 at 16k). Its expert-profile-driven hot cache
   pays off hard on RDNA4.
2. **Long-context decode flips to exllamav3.** exllamav3's decode is the flattest
   (75.4 → 50.0 at 128k, ~34% drop) and beats Polystrata at 16k (85.4 vs
   84.5) and from 64k on (61.1 vs 58.5 … 50.0 vs 43.2); Polystrata still leads at 32k
   (74.6 vs 67.8). Polystrata has the steepest slope (92.7 → 43.2, ~53% drop),
   consistent with its KV-heavy `--kv-whole` policy at long context.
3. **Prefill scaling:** exllamav3 holds ~1.2–1.4k tok/s out to 128k (1 214),
   better than anyone else there; Polystrata degrades from 1 963 (16k) to 710
   (128k); llama.cpp is the slowest prefill measured (966 at 4k, 1 032 at 16k only
   two lengths; MTP 824/881).
4. **MTP makes llama.cpp slower, not faster, in this stack.** With MTP the no-MTP
   baseline is beaten on prefill (824/881 vs 966/1 032 tok/s) and on decode
   (52.5 vs 62.7 tok/s short context; 50.8 vs 57.0 at 16k) — the CPU-side
   speculation cost outweighs draft acceptance at 30 CPU-resident experts.
5. **Hot cache benefit is real where it exists.** llama.cpp's 4 096 MiB
   `--moe-cache-mib` keeps decode within ~6 tok/s of the short-prompt numbers at
   4k/16k; exllamav3 sits at 74.9 at 4k (short: 75.4). Polystrata stays within
   ~9 tok/s through 16k before its long-context slope takes over.

## Caveats

- Two exllamav3 runs exist: 14:51Z and 15:59Z. Tables use the **15:59Z run**
  (DFlash2 draft config, full 128k coverage). The earlier 14:51Z run delivered
  ~35 tok/s short-context decode and rejected the 128k prompt — the ~2× decode
  gap between the two runs needs a written explanation before either can be
  called canonical.
- exllamav3 15:59Z TTFT is noisy at short prompts: 4k samples 3.4–8.1 s, 16k
  11.3–23.1 s, so its 670 tok/s (4k) prefill partially reflects that variance.
- Medians over ≤5 runs; a few runs ended early (`finish_reason: stop`) and are
  flagged in the JSON files; medians include them.
- Token-count checks (`prompt_tokens unchecked`) apply to all engines; benchmark
  prompts are synthetic corpus text.
- Each engine uses a different quantization (APEX GGUF vs EXL3 4.00 bpw), so
  cross-engine comparisons mix engine + format effects.

## Engine-specific benches (supplementary)

The exllamav3 recipe fork documents its own per-engine benchmarks, with scripts
in its `bench/`, methodology in `notes/measurements.md`, and patch notes in
`patches/README.md` (in [`willeyh-git/exl3-rdna4-recipe`](https://github.com/willeyh-git/exl3-rdna4-recipe)).
Highlights:

### ExLlamaV3: Ornith DFlash2 draft-token sweep

Median prefill/decode per draft-token count (three-repeat sweeps at ~1.5k, 7.7k,
15.9k, 32.2k prompt tokens; 128K context, Q8 main cache, Q4 draft cache,
`cpu_moe_split_experts: 100`; minus median draft acceptance):

![Ornith DFlash2 sweep](https://github.com/willeyh-git/exl3-rdna4-recipe/blob/main/bench/ornith_dflash2_benchmark.png?raw=true)

### Polystrata: cold benchmark (`polystrata.py benchmark`, 256 output tokens)

| Prompt target | Actual input tokens | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Cache hits | MTP right | RAM (GiB) | CPU % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short | 21 | 0.29 | 78 | 89.5 | 0.772 | 0.805 | 2.89 | 47.9 |
| 2K | 2 952 | 2.12 | 1 463 | 91.5 | 0.799 | 0.776 | 2.91 | 46.2 |
| 8K | 10 420 | 4.81 | 1 782 | 98.1 | 0.881 | 0.804 | 3.16 | 30.4 |
| 16K | 21 548 | 8.55 | 1 569 | 89.1 | 0.916 | 0.729 | 3.29 | 20.9 |
| 32K | 42 433 | 21.93 | 1 190 | 77.3 | 0.915 | 0.747 | 3.41 | 13.5 |

### Polystrata: interactive tool run (medians)

- Chat (three short questions, up to 100 new tokens): decode 96.9 tok/s, TTFT ~0.29 s.
- Fresh prompts: prefill 2 004 tok/s at ~4k, 2 225 at ~16k, 1 824 at ~32k.
- Repeated agent session (15 turns): decode 92.4 tok/s, 68.6% cache hits, ~249 re-read tokens per session, ~2.3 s turn TTFT.

Note: these use different prompt targets and shorter (256-token) outputs, so they
are a reference for each engine's caching behavior — most notably Polystrata's
cache-hit rate climbing with context length — not a controlled head-to-head with
the llm-bench protocol above.

## Raw results

- `exllamav3-20261111T155934Z-ornith-1-5-35b-a3b-exl3-4-00bpw-24gb.json` (canonical)
- `exllamav3-20261111T145111Z-ornith-1-5-35b-a3b-exl3-4-00bpw-24gb.json` (stale run)
- `llamacpp-ornith-1-5-35b-a3b-apex-i-quality-no-mtp.json`
- `llamacpp-ornith-1-5-35b-a3b-apex-i-quality-mtp.json`
- `polystrata-ornith-1-5-35b-a3b-apex-i-quality.json`