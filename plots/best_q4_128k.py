#!/usr/bin/env python3
"""Winners per model across context length: prefill on top, decode below.

Same style as the per-model plots, but each model contributes only its
winning combo (highest total prefill@128k + decode@128k among runs that
reached 128k). No other engines are plotted.

Outputs: plots/best_q4_128k.png
Run: uv run --with matplotlib python best_q4_128k.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# short-context decode, then 4k..128k
decode_x = [0.02, 4, 16, 32, 64, 96, 128]
prefill_x = [4, 16, 32, 64, 96, 128]
decode_ticks = ["~16/90/243", "4k", "16k", "32k", "64k", "96k", "128k"]

RUNS = [
    {
        # Ornith winner: exllamav3 EXL3 4.0 bpw + DFlash2 (canonical 20:53Z)
        "label": "Ornith 1.5 35B-A3B — EXL3 4.0 bpw + DFlash2 — exllamav3 (20:53Z)",
        "color": "#d62728",
        "prefill": [1293, 1356, 1344, 1279, 1222, 1168],
        "decode": [94.0, 92.3, 104.5, 82.6, 73.1, 61.7, 57.3],
    },
    {
        # Qwen3.6 winner: EXL3 4.0 bpw + DFlash2, 2 drafts (11:44Z aside)
        "label": "Qwen3.6-35B-A3B — EXL3 4.0 bpw + DFlash2 (2 drafts) — exllamav3 (11:44Z)",
        "color": "#1f77b4",
        "prefill": [1463.0, 1534.3, 1492.1, 1398.3, 1312.4, 1245.6],
        "decode": [105.4, 107.0, 98.9, 93.0, 86.3, 80.1, 71.6],
    },
    {
        # QFN Coder (only run): Polystrata nov-7, --spec 2
        "label": "Qwen3.8-Flash-Next Coder — IQ1_M — Polystrata nov-7, --spec 2",
        "color": "#2ca02c",
        "prefill": [1186.1, 1465.5, 1520.0, 1554.3, 1543.6, 1571.6],
        "decode": [68.1, 44.6, 46.2, 46.0, 45.1, 45.0, 44.0],
    },
]


def main():
    fig, (ax_pre, ax_dec) = plt.subplots(2, 1, figsize=(8.5, 8.6), sharex=True)

    for r in RUNS:
        ax_pre.plot(prefill_x, r["prefill"], marker="o", color=r["color"],
                    lw=1.8, ms=5, label=r["label"])
        ax_dec.plot(decode_x, r["decode"], marker="o", color=r["color"],
                    lw=1.8, ms=5, label=r["label"] + " (median)")

    ax_pre.set_title("Prefill throughput (tok/s)")
    ax_pre.grid(True, alpha=0.25)
    ax_pre.legend(loc="lower right", fontsize=8, framealpha=0.9)

    ax_dec.set_title("Decode throughput (tok/s, median)")
    ax_dec.grid(True, alpha=0.25)
    ax_dec.set_xticks(decode_x, decode_ticks)
    ax_dec.set_xlabel("Prompt / context length")
    ax_dec.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Best Q4-class combo per model\n(highest total prefill + decode @128k)",
        y=0.995,
    )
    out = os.path.join(os.path.dirname(__file__), "best_q4_128k.png")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()