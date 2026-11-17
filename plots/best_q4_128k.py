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
        # Ornith winner: exllamav3 EXL3 4.0 bpw, tweaked no-specish baseline
        # (FULL RUN 13:55Z; 96k/128k decode on 2 runs each — see
        # ornith-1-5-tweaks.md). Beats old DFlash2 winner 1168 + 57.3 = 1725
        # by 1924.5 + 55.7 = 2480.
        "label": "Ornith 1.5 35B-A3B — EXL3 4.0 bpw, no-spec tweaked — exllamav3 (13:55Z)",
        "color": "#d62728",
        "prefill": [2293.7, 2405.7, 2473.1, 2310.5, 1998.1, 1924.5],
        "decode": [91.8, 93.8, 99.6, 86.3, 72.7, 59.7, 55.7],
    },
    {
        # Qwen3.6 winner: EXL3 4.0 bpw, no-spec, tweaked baseline (12:59Z FULL RUN)
        "label": "Qwen3.6-35B-A3B — EXL3 4.0 bpw, no-spec — exllamav3 (tweaked, 12:59Z)",
        "color": "#1f77b4",
        "prefill": [2314, 2566, 2606, 2411, 2178, 2009],
        "decode": [109.4, 110.0, 103.9, 101.3, 93.2, 89.5, 82.8],
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