#!/usr/bin/env python3
"""KAT-Coder V2.5 Dev (EXL3 5.0 bpw hq) — prefill & decode context curves.

Numbers from the two full-range runs:
  baseline  20261113T150956Z (tag "13GB vram" — serve defaults)
  tweaked   20261117T113004Z (chunk 5120, output_chunking, vision offload,
                cpu_stage_threads 6, threads 7, MTP 2, cpu_moe_split_experts 190)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

short_ctx = ["Short", "4k", "16k", "32k", "64k", "96k", "128k"]
pre_ctx = ["4k", "16k", "32k", "64k", "96k", "128k"]

decode = {
    "Baseline '13 GB vram' (11-13)": [54.4, 61.1, 56.3, 59.1, 52.9, 52.9, 51.0],
    "Chunk 5120 + cpu_stage_threads 6 (11-17)": [66.7, 73.1, 65.4, 64.9, 60.9,
                                                  57.0, 53.6],
}
prefill = {
    "Baseline '13 GB vram' (11-13)": [1229, 1285, 1313, 1283, 1230, 1175],
    "Chunk 5120 + cpu_stage_threads 6 (11-17)": [1284, 1349, 1375, 1324, 1261,
                                                 1203],
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
for label, vals in decode.items():
    ax1.plot(short_ctx, vals, "o-", label=label)
    for x, y in zip(short_ctx, vals):
        ax1.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=7.5)
ax1.set_title("Decode tokens/sec vs context (window median)")
ax1.set_ylabel("tokens/sec")
ax1.grid(alpha=0.3)
ax1.legend(fontsize=8)

for label, vals in prefill.items():
    ax2.plot(pre_ctx, vals, "s-", label=label)
    for x, y in zip(pre_ctx, vals):
        ax2.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=7.5)
ax2.set_title("Prefill tokens/sec by prompt length")
ax2.set_ylabel("tokens/sec")
ax2.grid(alpha=0.3)
ax2.legend(fontsize=8)

plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "kat-coder-5.00bpw.png"), dpi=150)
print("saved kat-coder-5.00bpw.png")