#!/usr/bin/env python3
"""Nemotron 3.5 Lightning 30B-A3B (EXL3 3.25 bpw) — prefill & decode curves.

Numbers from the 2026-11-17 15:12Z full run (max_context 153600):
  offload experts 5, chunk 4096, Q8 cache.
The prior same-day run (14:19Z) stopped short of 128k (context cap) and is
excluded from the docs.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

short_ctx = ["Short", "4k", "16k", "32k", "64k", "96k", "128k"]
pre_ctx = ["4k", "16k", "32k", "64k", "96k", "128k"]

decode = [102.6, 102.3, 101.1, 96.8, 88.9, 82.8, 78.8]
prefill = [3380, 3758, 3627, 3400, 3163, 2973]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
ax1.plot(short_ctx, decode, "o-")
for x, y in zip(short_ctx, decode):
    ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                 xytext=(0, 7), ha="center", fontsize=8)
ax1.set_title("Decode tokens/sec vs context (window median)")
ax1.set_ylabel("tokens/sec")
ax1.grid(alpha=0.3)
ax1.text(0.5, 0.06, "96k/128k rest on 2 runs each", transform=ax1.transAxes,
         ha="center", fontsize=8, style="italic")

ax2.plot(pre_ctx, prefill, "s-")
for x, y in zip(pre_ctx, prefill):
    ax2.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                 xytext=(0, 7), ha="center", fontsize=8)
ax2.set_title("Prefill tokens/sec by prompt length")
ax2.set_ylabel("tokens/sec")
ax2.grid(alpha=0.3)

plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "nemotron-3.5-lightning-3.25bpw.png"), dpi=150)
print("saved nemotron-3.5-lightning-3.25bpw.png")