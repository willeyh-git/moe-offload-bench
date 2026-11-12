#!/usr/bin/env python3
"""Best Q4-class engine/combo per model at 128k (README figure).

Score = prefill@128k + decode@128k (tok/s); only runs that reached a 128k
prompt are eligible. The highest total per model wins.

Outputs: plots/best_q4_128k.png
Run: uv run --with matplotlib python best_q4_128k.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = {
    "Ornith 1.5 35B-A3B": {
        "combos": [
            {"name": "llama.cpp\nno-MTP", "pre": 831.0, "dec": 35.0},
            {"name": "llama.cpp\nMTP", "pre": 719.0, "dec": 32.1},
            {"name": "exllamav3\nEXL3 4.0 + DFlash2", "pre": 1168.0, "dec": 57.3},
            {"name": "Polystrata\nAPEX-I-QUALITY", "pre": 709.0, "dec": 41.1},
        ],
    },
    "Qwen3.6-35B-A3B": {
        "combos": [
            {"name": "exllamav3\nEXL3 4.0 + MTP (2)", "pre": 1211.2, "dec": 83.3},
            {"name": "exllamav3\nEXL3 4.0 + DFlash2 (2)", "pre": 1245.6, "dec": 71.6},
            {"name": "Polystrata\nUD Q4_K_XL", "pre": 733.7, "dec": 62.0},
            {"name": "Strata\nUD Q4_K_XL", "pre": 397.9, "dec": 9.3},
        ],
    },
    "Qwen3.8-Flash-Next Coder": {
        "combos": [
            {"name": "Polystrata nov-7\nIQ1_M, --spec 2", "pre": 1571.6, "dec": 44.0},
        ],
    },
}

# llama.cpp Qwen3.6 runs (32k max) and Qwen3.6 short-range runs (32k max)
# are excluded: the run must reach 128k.


def main():
    fig, axes = plt.subplots(
        1, 3, figsize=(15.5, 5.2), constrained_layout=True
    )
    for ax, (model, data) in zip(axes, MODELS.items()):
        combos = data["combos"]
        winner = max(combos, key=lambda c: c["pre"] + c["dec"])
        x = np.arange(len(combos))
        w = 0.38
        pre = [c["pre"] for c in combos]
        dec = [c["dec"] for c in combos]
        totals = [c["pre"] + c["dec"] for c in combos]

        is_win = [c is winner for c in combos]
        pre_b = ax.bar(
            x - w / 2, pre, w,
            color=["tab:blue" if w_ else "#c7d3ea" for w_ in is_win],
            label="Prefill @128k",
        )
        dec_b = ax.bar(
            x + w / 2, dec, w,
            color=["tab:orange" if w_ else "#f2e0cf" for w_ in is_win],
            label="Decode @128k",
        )
        for i, (c, tot) in enumerate(zip(combos, totals)):
            # annotate the total above the taller of the two bars
            top = max(pre[i], dec[i])
            mark = "  WINNER" if i == [j for j, c2 in enumerate(combos) if c2 is winner][0] else ""
            ax.annotate(
                f"{tot:,.0f}{mark}",
                (x[i], top),
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold" if mark else "normal",
                color="#1a1a2e" if mark else "#8a8a99",
            )
        ax.set_title(model, fontweight="bold")
        ax.set_xticks(x, [c["name"] for c in combos], fontsize=8)
        ax.set_ylabel("tok/s (medians)")
        ax.set_ylim(0, max(max(pre), 1100) * 1.25)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle(
        "Best Q4-class combo per model at 128k — total = prefill + decode (tok/s)",
        fontsize=12,
    )
    out = os.path.join(os.path.dirname(__file__), "best_q4_128k.png")
    fig.savefig(out, dpi=150)
    print("saved", out)
    for model, data in MODELS.items():
        w_ = max(data["combos"], key=lambda c: c["pre"] + c["dec"])
        print(f"{model}: winner = {w_['name'].replace(chr(10), ' ')} "
              f"({w_['pre']:,.0f} + {w_['dec']:.1f} = {w_['pre'] + w_['dec']:,.1f})")


if __name__ == "__main__":
    main()