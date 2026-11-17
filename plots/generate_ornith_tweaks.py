"""Comparison graphs for the Ornith 1.5 35B-A3B 2026-11-15 tweaked runs.

Two runs, both "FULL RUN" on the tuned exllamav3 baseline (chunk 5120,
vision offload, pinned arena false, cpu_stage_threads 6, threads 7):
  - Run 1 (13:55Z): tag says no spec; config had no draft option — draft
    status unverified (see ornith-1-5-tweaks.md).
  - Run 2 (14:31Z): tag "FULL RUN dflash2 num tokens 4" — config pointed at
    the DFlash2 drafter (draft_model_name), 4 draft tokens.

The 20:53Z run (pre-tune exllamav3, DFlash2 4-draft) is included from its
JSON (results/ornith/exllamav3-20261111T205345Z-...json) as a reference line.

JSON context key "126000" (128 987 prompt tokens) is the 128k target and is
labeled 128k below.

Outputs: results/ornith-tweaks/01..04 PNGs.
Run: uv run --with matplotlib python plots/generate_ornith_tweaks.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/home/willyhetland/Storage/code/llm-bench/results"
OUTPUT_DIR = os.path.abspath(os.path.join(HERE, "..", "results", "ornith-tweaks"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

REV = "exl3-4-00bpw-24gb"
path_run1 = f"{SRC}/20261115T135554Z-ornith-1-5-35b-a3b-{REV}.json"
path_run2 = f"{SRC}/20261115T143144Z-ornith-1-5-35b-a3b-{REV}.json"
path_ref = os.path.abspath(os.path.join(HERE, "..", "results", "ornith",
                                        f"exllamav3-20261111T205345Z-ornith-1-5-35b-a3b-{REV}.json"))

runs = [json.load(open(p)) for p in (path_run1, path_run2)]
ref = json.load(open(path_ref))
short_labels = ["no-spec (13:55Z)", "DFlash2 4 drafts (14:31Z)"]

run_colors = ["#4e79a7", "#e15759"]
bar_colors = ["#8172b3", "#cc4c99", "#4e79a7"]  # 4k / 16k / 32k prompt bars
ctx_keys = ["short", "4000", "16000", "32000", "64000", "96000", "126000"]
ctx_labels = ["Short", "4k", "16k", "32k", "64k", "96k", "128k"]


def decode_meds(r):
    out = [r["decode"]["tps_median"]]
    for k in ctx_keys[1:]:
        out.append(r["decode_at_context"][k]["tps_median"])
    return out


def ref_meds(r):
    # 20:53Z schema: "decode" short + "decode_at_context" 4k..126000
    if "decode_at_context" in r and "short" not in r.get("decode", {}):
        out = [r["decode"]["tps_median"]]
        for k in ctx_keys[1:]:
            out.append(r["decode_at_context"][k]["tps_median"])
        return out
    # fallback: hard-coded canonical numbers from ORNITH1.5-35B-A3B.md
    return [94.0, 92.3, 104.5, 82.6, 73.1, 61.7, 57.3]


# --- Graph 1: Prefill TPS ---
fig, ax = plt.subplots(figsize=(8, 5))
width = 0.2
pre_keys = ["4000", "16000", "32000", "64000", "96000", "126000"]
pre_labels = ["4k", "16k", "32k", "64k", "96k", "128k"]
xx = range(len(runs))
for i, key in enumerate(["4000", "16000", "32000"]):
    vals = [r["prefill"][key]["tps"] for r in runs]
    ax.bar([j + (i - 1) * width for j in xx], vals, width,
           label=f"{pre_labels[i]} prompt", color=bar_colors[i], alpha=0.85)
ax.set_ylabel("Tokens/sec (window)")
ax.set_title("Prefill Performance")
ax.set_xticks(list(xx))
ax.set_xticklabels(short_labels, fontsize=9)
ax.legend()
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "01-prefill-tps.png"), dpi=150)
plt.close(fig)

# --- Graph 2: Decode TPS by context (run1, run2, pre-tune reference) ---
fig, ax = plt.subplots(figsize=(10, 5))
w = 0.26
xx = range(len(ctx_keys))
ax.bar([j - w for j in xx], ref_meds(ref), w, label="pre-tune 20:53Z", color="#999999")
ax.bar([j for j in xx], decode_meds(runs[0]), w, label=short_labels[0], color=run_colors[0])
ax.bar([j + w for j in xx], decode_meds(runs[1]), w, label=short_labels[1], color=run_colors[1])
ax.set_xticks(list(xx))
ax.set_xticklabels(ctx_labels)
ax.set_ylabel("Tokens/sec (median)")
ax.set_title("Decode TPS by Context Length")
ax.legend()
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "02-decode-context.png"), dpi=150)
plt.close(fig)

# --- Graph 3: short-context decode bars ---
fig, ax = plt.subplots(figsize=(8, 5))
sh = [r["decode"]["tps_median"] for r in runs]
ax.bar(list(range(len(runs))), sh, 0.5, color=run_colors)
ax.axhline(ref_meds(ref)[0], color="#999999", ls="--", lw=1.5,
           label=f"pre-tune 20:53Z ({ref_meds(ref)[0]:.1f})")
for j, v in enumerate(sh):
    ax.text(j, v + 1, f"{v:.1f}", ha="center", fontsize=9)
ax.set_xticks(list(range(len(runs))))
ax.set_xticklabels(short_labels, fontsize=9)
ax.set_ylabel("Tokens/sec (window)")
ax.set_title("Decode TPS — Short Context")
ax.legend()
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "03-decode-sh.png"), dpi=150)
plt.close(fig)

# --- Graph 4: Prefill TTFT bars per length ---
fig, ax = plt.subplots(figsize=(10, 5))
w = 0.35
xx = range(len(pre_keys))
ax.bar([j - w / 2 for j in xx],
       [runs[0]["prefill"][k]["ttft_seconds"] for k in pre_keys], w,
       label=short_labels[0], color=run_colors[0])
ax.bar([j + w / 2 for j in xx],
       [runs[1]["prefill"][k]["ttft_seconds"] for k in pre_keys], w,
       label=short_labels[1], color=run_colors[1])
ax.set_xticks(list(xx))
ax.set_xticklabels(pre_labels)
ax.set_ylabel("Time to First Token (seconds)")
ax.set_title("Prefill Latency (TTFT)")
ax.legend()
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "04-prefill-ttft.png"), dpi=150)
plt.close(fig)

print("Generated graphs in", OUTPUT_DIR)