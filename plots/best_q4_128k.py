#!/usr/bin/env python3
"""Winners per model at 128k: one graph for prefill, one for decode.

Only the winning combo per model (highest prefill@128k + decode@128k total,
eligible runs must reach 128k) is plotted. Legend: model / quant / engine.

Outputs: plots/best_q4_128k.png
Run: uv run --with matplotlib python best_q4_128k.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WINNERS = [
    {
        "model": "Ornith 1.5 35B-A3B",
        "legend": "Ornith 1.5 35B-A3B\nEXL3 4.0 bpw, DFlash2\nexllamav3",
        "pre": 1168.0,
        "dec": 57.3,
    },
    {
        "model": "Qwen3.6-35B-A3B",
        "legend": "Qwen3.6-35B-A3B\nEXL3 4.0 bpw, DFlash2 (2 drafts)\nexllamav3",
        "pre": 1245.6,
        "dec": 71.6,
    },
    {
        "model": "Qwen3.8-Flash-Next Coder",
        "legend": "Qwen3.8-Flash-Next Coder\nIQ1_M\nPolystrata nov-7 --spec 2",
        "pre": 1571.6,
        "dec": 44.0,
    },
]
COLORS = ["tab:blue", "tab:orange", "tab:green"]


def main():
    fig, (ax_pre, ax_dec) = plt.subplots(
        1, 2, figsize=(10.5, 5.0), constrained_layout=True
    )
    x = np.arange(len(WINNERS))
    model_labels = [w["model"] for w in WINNERS]

    for ax, key, title in (
        (ax_pre, "pre", "Prefill @ 128k"),
        (ax_dec, "dec", "Decode @ 128k"),
    ):
        vals = [w[key] for w in WINNERS]
        bars = ax.bar(x, vals, width=0.6, color=COLORS)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x, model_labels, fontsize=9, rotation=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(vals) * 1.2)
        for b, v in zip(bars, vals):
            label = f"{v:,.0f}" if key == "pre" else f"{v:.1f}"
            ax.annotate(
                label,
                (b.get_x() + b.get_width() / 2, v),
                ha="center",
                va="bottom",
                fontsize=9.5,
            )

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=c)
        for c in COLORS
    ]
    ax_pre.legend(
        handles, [w["legend"] for w in WINNERS],
        loc="upper left", fontsize=8.5, framealpha=0.9,
    )
    fig.suptitle(
        "Winner per model at 128k (highest total prefill + decode; Q4-class quants)",
        fontsize=12,
    )
    out = os.path.join(os.path.dirname(__file__), "best_q4_128k.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()