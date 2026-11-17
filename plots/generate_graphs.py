#!/usr/bin/env python3
"""Generate benchmark comparison graphs for Qwen3.6-35B-A3B runs."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

RESULTS_DIR = "/home/willyhetland/Storage/code/llm-bench/results"
OUTPUT_DIR = "/home/willyhetland/Storage/code/moe-offload-bench/plots"

# Load all Qwen3.6-35B-A3B tweak runs (2026-11-15 bake-off + 2026-11-17 batch 6144)
files = sorted([
    f for f in os.listdir(RESULTS_DIR)
    if f.startswith(("20261115", "20261117")) and "qwen3-6-35b" in f
])  # ISO timestamp prefix -> chronological, oldest first

runs = []
for f in files:
    with open(os.path.join(RESULTS_DIR, f)) as fh:
        data = json.load(fh)
    runs.append(data)

# Extract key metrics
labels = [r["tag"] for r in runs]
short_labels = [
    "batch inc\nno cpu_threads",
    "batch 4096\npinned false",
    "pinned true",
    "batch 4096\ncpu_threads 6",
    "batch 5120\ncpu_threads 6",
    "swizzle -1",
    "wide -1",
    "gdn_sub 2048",
    "bc_gdn -1",
    "FULL RUN\n5120",
    "FULL RUN\n6144",
]

prefill_4k = [r["prefill"]["4000"]["tps"] for r in runs]
prefill_16k = [r["prefill"]["16000"]["tps"] for r in runs]
prefill_32k = [r["prefill"]["32000"]["tps"] for r in runs]

decode_med = [r["decode"]["tps_median"] for r in runs]
decode_4k_med = [r["decode_at_context"]["4000"]["tps_median"] for r in runs]
decode_16k_med = [r["decode_at_context"]["16000"]["tps_median"] for r in runs]
decode_32k_med = [r["decode_at_context"]["32000"]["tps_median"] for r in runs]

decode_4k_min = [r["decode_at_context"]["4000"]["tps_min"] for r in runs]
decode_4k_max = [r["decode_at_context"]["4000"]["tps_max"] for r in runs]
decode_16k_min = [r["decode_at_context"]["16000"]["tps_min"] for r in runs]
decode_16k_max = [r["decode_at_context"]["16000"]["tps_max"] for r in runs]
decode_32k_min = [r["decode_at_context"]["32000"]["tps_min"] for r in runs]
decode_32k_max = [r["decode_at_context"]["32000"]["tps_max"] for r in runs]

x = range(len(runs))
w = 0.25

# --- Graph 1: Prefill TPS ---
fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar([i - w for i in x], prefill_4k, w, label="4k prompt", color="#4e79a7")
bars2 = ax.bar(x, prefill_16k, w, label="16k prompt", color="#f28e2b")
bars3 = ax.bar([i + w for i in x], prefill_32k, w, label="32k prompt", color="#e15759")
ax.set_ylabel("Tokens/sec")
ax.set_title("Prefill Performance — Qwen3.6-35B-A3B-exl3-4.00bpw")
ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=8)
ax.legend()
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "01-prefill-tps.png"), dpi=150)
plt.close(fig)

# --- Graph 2: Decode TPS at context ---
fig, ax = plt.subplots(figsize=(10, 5))
ctx_labels = ["No ctx", "4k", "16k", "32k"]
ctx_medians = [decode_med, decode_4k_med, decode_16k_med, decode_32k_med]
ctx_min = [decode_med, decode_4k_min, decode_16k_min, decode_32k_min]
ctx_max = [decode_med, decode_4k_max, decode_16k_max, decode_32k_max]

colors = ["#4e79a7", "#f28e2b", "#e15759", "#59a14f"]
for i, (med, lo, hi) in enumerate(zip(ctx_medians, ctx_min, ctx_max)):
    down = [med[j] - lo[j] for j in x]
    up = [hi[j] - med[j] for j in x]
    ax.errorbar(x, med, yerr=[down, up],
                fmt="o", label=ctx_labels[i], color=colors[i], capsize=3, markersize=6)

ax.set_ylabel("Tokens/sec (window)")
ax.set_title("Decode Performance at Different Context Lengths")
ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=8)
ax.legend()
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "02-decode-context.png"), dpi=150)
plt.close(fig)

# --- Graph 3: Decode TPS bar chart ---
fig, ax = plt.subplots(figsize=(10, 5))
ctx_bars = [decode_med, decode_4k_med, decode_16k_med, decode_32k_med]
ctx_names = ["No context", "4k context", "16k context", "32k context"]
for i, (vals, name) in enumerate(zip(ctx_bars, ctx_names)):
    ax.bar([j + i * 0.2 for j in x], vals, 0.2, label=name, color=colors[i])

ax.set_ylabel("Tokens/sec (window)")
ax.set_title("Decode Performance — Bar Comparison")
ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=8)
ax.legend()
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "03-decode-bars.png"), dpi=150)
plt.close(fig)

# --- Graph 4: Prefill TTFT ---
fig, ax = plt.subplots(figsize=(10, 5))
ttft_4k = [r["prefill"]["4000"]["ttft_seconds"] for r in runs]
ttft_16k = [r["prefill"]["16000"]["ttft_seconds"] for r in runs]
ttft_32k = [r["prefill"]["32000"]["ttft_seconds"] for r in runs]

ax.plot(x, ttft_4k, "o-", label="4k prompt", color="#4e79a7", linewidth=2, markersize=8)
ax.plot(x, ttft_16k, "s-", label="16k prompt", color="#f28e2b", linewidth=2, markersize=8)
ax.plot(x, ttft_32k, "^-", label="32k prompt", color="#e15759", linewidth=2, markersize=8)

ax.set_ylabel("Time to First Token (seconds)")
ax.set_title("Prefill Latency (TTFT)")
ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=8)
ax.legend()
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "04-prefill-ttft.png"), dpi=150)
plt.close(fig)

print("Generated graphs:")
for f in ["01-prefill-tps.png", "02-decode-context.png", "03-decode-bars.png", "04-prefill-ttft.png"]:
    print(f"  {OUTPUT_DIR}/{f}")