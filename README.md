# Exploring MoE CPU offloading and hot cache experts

Exploring MoE expert CPU offloading and hot expert caches across different
inference engines on RDNA4 (Radeon 9070 XT).

## Goal

Run MoE models that do **not** fit in the GPU's 16 GB of VRAM at **at least
128K context**, with **prefill over 1 500 tok/s** and **decode over 100 tok/s**.
The idea is to offload some experts to CPU with as small a performance hit as
possible — it will obviously never match a fully GPU-resident model, but the
goal is to stay in the same ballpark. Everything here is on the lesser-optimized
AMD ROCm stack, where the gap between systems is widest.

Decisions already made along the way:

- Main KV cache at Q8 to leave VRAM headroom; draft cache at Q4.
- CPU-resident experts are the offload unit (llama.cpp `--n-cpu-moe`, exllamav3
  `cpu_moe_split_experts`), with per-engine hot-expert caches where available.

### Notable findings so far

- **Polystrata/Strata pegs the GPU hardest** — it runs the 9070 XT at its
  power limit (300 W), while llama.cpp and exllamav3 only reach ~150–200 W.
  Despite that, the CPU-side decode speed is not noticeably worse than on
  Polystrata/Strata, so llama.cpp/exllamav3 deliver the best absolute
  watts-per-token.
- **On the original Strata engine, throughput degrades badly as context
  grows** for the heavier models (Ornith Polystrata prefill falls 1 723 →
  709 tok/s from 4k→128k; Strata Unsloth Qwen3.6 decode collapses 59.4 → 9.3
  tok/s over the same range). The Flash Next Coder keeping a consistent ~1 550
  tok/s prefill all the way to 128k is quite impressive in comparison.
- **Cache tweaks help only modestly**: llama.cpp's 4 096 MiB hot-expert cache
  (`--moe-cache-mib`) gives small gains (decode holds to ~16k before falling
  off), and the CPU-decode share flag (`--moe-cache-cpu-share`) did not
  improve decode noticeably — numbers in [moe-cache.md](moe-cache.md).

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

## Best ~Q4 engine combo per model at 128k

Same layout as the per-model plots — prefill on top, decode below — with one
line per model: the winning combo only (highest total of prefill + decode at
128k, among runs that reached 128k). No other engines are plotted.
Q4-class quants for Ornith (APEX GGUF / EXL3 4.00 bpw) and Qwen3.6 (EXL3 4.00
bpw / UD Q4_K_XL GGUF); QFn Coder is IQ1_M, included for reference.

![Best Q4-class combo per model at 128k](plots/best_q4_128k.png)

| Model | Winner (quant / engine) | Prefill @128k | Decode @128k |
| --- | --- | ---: | ---: |
| Ornith 1.5 35B-A3B | EXL3 4.0 bpw, no-spec / exllamav3 (tweaked)† | **1 925 tok/s** | **55.7 tok/s** |
| Qwen3.6-35B-A3B | EXL3 4.0 bpw, MTP (2), chunk 5120 / exllamav3 (tweaked) | **2 009 tok/s** | **82.8 tok/s** |
| Qwen3.6-35B-A3B (6.00 bpw best combo)†† | EXL3 6.0 bpw, MTP (2), chunk 6144 / exllamav3 (tweaked) | 1 837 tok/s | 65.1 tok/s |
| Qwen3.8-Flash-Next Coder | IQ1_M / Polystrata nov-7 `--spec 2` | 1 572 tok/s | 44.0 tok/s |
| KAT-Coder V2.5 Dev (MTP) | EXL3 5.0 bpw (hq), MTP (2), chunk 5120 / exllamav3 (tweaked)‡ | **1 203 tok/s** | **53.6 tok/s** |
| Nemotron 3.5 Lightning 30B-A3B | EXL3 3.25 bpw, Q8 cache, no-spec / exllamav3§ | **2 973 tok/s** | **78.8 tok/s** |

Excluded for not reaching 128k: llama.cpp Qwen3.6 runs and the exllamav3
short-range Qwen3.6 runs (~32k max). QFn Coder has a single eligible run, so
it is its own winner by default.

The 2026-11-17 Qwen3.6 chunk-6144 re-run (MTP 2, otherwise identical to the
winner) lands at 1 998 / 81.6 at 128k — within ~2% of the chunk-5120 winner on
both prefill and decode, so it does not take the win (128k total 2 079 vs
2 092). The same tweaked baseline was also run at EXL3 5.0 bpw twice (2026-11-17:
17:37Z, tag says “batch 5120” but the run used chunk 6144, and 18:06Z at a
genuine chunk 5120): 1 955 / 75.0 (total 2 029.6) and 1 861 / 75.5 (total
1 936.5) at 128k — not wins either (chunk 5120 decodes better at 32k–96k,
chunk 6144 prefills better). The tweaked baseline was also run at 6.00 bpw
(18:47Z chunk 5120, 19:01Z chunk 6144): 1 722 / 70.7 (total ≈ 1 793) and
1 837 / 65.1 (total ≈ 1 903) — 6.00 bpw loses ~15–21% more decode than the
4.0 bpw winner and the Q6 128k row above is only the best Q6 combo, not a
challenger for the Qwen3.6 win. See
[Qwen3.6-35b-a3b-improvements.md](Qwen3.6-35b-a3b-improvements.md) and
[QWEN3.6-35B-A3B.md](QWEN3.6-35B-A3B.md).

† The Ornith tweaked run's 96k/128k decode medians rest on 2 runs each, and its
tag doesn't confirm spec was off at runtime (see
[ornith-1-5-tweaks.md](ornith-1-5-tweaks.md)). It replaces the previous DFlash2
winner (1 168 / 57.3 at 128k) on the prefill+decode total (2 480 vs 1 725);
short- and mid-context decode is roughly flat vs that run.

‡ KAT-Coder's 96k/128k decode medians rest on 2 runs each, and
`cpu_moe_split_experts: 190` is the OOM floor (anything lower does not start);
see [KAT-Coder.md](KAT-Coder.md) for the run and the vs-baseline comparison.

§ Nemotron 3.5 Lightning's 96k/128k decode medians rest on 2 runs each; the
prior same-day run (2026-11-17 14:19Z) stopped short of 128k, so the winner
above is the 15:12Z run with max_context 153 600 (see
[NEMOTRON3.5-LIGHTNING.md](NEMOTRON3.5-LIGHTNING.md)).

†† Q6 row: best of the two tweaked Q6 full runs (18:47Z chunk 5120: 1 722 / 70.7;
19:01Z chunk 6144: 1 837 / 65.1, 155 experts offloaded, ~14.3–14.7 GB VRAM),
96k/128k decode medians rest on 2 runs each. Neither approaches the Qwen3.6
winner (totals ≈ 1 793 / 1 903 vs 2 092) — 6.00 bpw mainly loses decode; see
the 6.00 bpw section of [QWEN3.6-35B-A3B.md](QWEN3.6-35B-A3B.md).

Generated with [`plots/best_q4_128k.py`](plots/best_q4_128k.py)
(`uv run --with matplotlib python best_q4_128k.py`).

## Raw results (master list)

JSON files live under `results/`, grouped by model. Full annotations are in the
per-model files.

### `results/ornith/` — Ornith 1.5 35B-A3B (11 files)

See [ORNITH1.5-35B-A3B.md](ORNITH1.5-35B-A3B.md) — canonical, superseded and
stale runs annotated there.

### `results/qwen/` — Qwen3.6-35B-A3B (32 files)

- `20261112T155239Z-qwen3-6-35b-a3b-ud-q4-k-xl.json` (UD Q4_K_XL, PolyStrata)
- `20261112T151400Z-qwen3-6-35b-a3b-ud-q4-k-xl.json` (UD Q4_K_XL, Strata Unsloth build)
- `20261112T134157Z-qwen-3-6-q4-llama.json` (llama.cpp no-drafter)
- `20261112T132841Z-qwen-3-6-q4-llama.json` (llama.cpp MTP)
- 15 × `exllamav3-20261112T…-q{4,5,6}-exl3-*.json` / `20261117T…Z-q6-exl3-6-00bpw-hb6.json`
  (EXL3 4.00/5.00/6.00 bpw; no-spec / MTP / DFlash2 short-range runs plus
  2-draft asides, and the two 2026-11-17 Q6 tweaked full runs — chunk 5120
  18:47Z, chunk 6144 19:01Z)
- 11 × `20261115T…Z-…-exl3-4-00bpw.json` / `20261117T095948Z-…json` (tweak-bake-off runs
  0–9 from 2026-11-15 plus the 2026-11-17 chunk-6144 re-run — see
  [Qwen3.6-35b-a3b-improvements.md](Qwen3.6-35b-a3b-improvements.md) for tags)
- `20261117T173715Z-qwen3-6-35b-a3b-exl3-5-00bpw.json` (tweak baseline at EXL3 5.0 bpw,
  full range 4k→128k; tag says “batch 5120” but the run used chunk 6144 —
  [QWEN3.6-35B-A3B.md](QWEN3.6-35B-A3B.md), 5.00 bpw section)
- `20261117T180639Z-qwen3-6-35b-a3b-exl3-5-00bpw.json` (tweak baseline at EXL3 5.0 bpw,
  genuine chunk 5120, full range 4k→128k — same md)

See [QWEN3.6-35B-A3B.md](QWEN3.6-35B-A3B.md) for the full annotated list.

### `results/qwen-flash-next/` — Qwen3.8-Flash-Next (1 file)

- `polystrata-20261112T171930Z-qwen3-8-flash-next-coder-iq1-m.json` (Coder IQ1_M, Polystrata nov-7 build, `--spec 2` — [QWEN3.8-FLASH_NEXT.md](QWEN3.8-FLASH_NEXT.md))

### `results/kat-coder/` — KAT-Coder V2.5 Dev (2 files)

- `20261117T113004Z-kat-coder-v2-5-dev-mtp-exl3-5bpw-hq.json` (EXL3 5.0 bpw hq, MTP 2, chunk 5120 + cpu_stage_threads 6, `cpu_moe_split_experts` 190 — the tweaked full run, [KAT-Coder.md](KAT-Coder.md))
- `20261113T150956Z-kat-coder-v2-5-dev-mtp-exl3-5bpw-hq.json` ("13 GB vram" baseline full run, same md)

### `results/nemotron/` — Nemotron 3.5 Lightning 30B-A3B (1 file)

- `20261117T151238Z-nemotron-3-5-lightning-exl3-3-25.json` (EXL3 3.25 bpw,
  expert offload 5, chunk 4096, Q8 cache, max_context 153 600 —
  [NEMOTRON3.5-LIGHTNING.md](NEMOTRON3.5-LIGHTNING.md)); the 14:19Z attempt
  that stopped before 128k is intentionally not copied