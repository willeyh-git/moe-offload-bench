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
        # Qwen3.6 winner: EXL3 4.0 bpw, MTP (2 draft tokens, no external
        # drafter) on the tweaked baseline, chunk 5120 (12:59Z FULL RUN)
        # 128k total 2 009 + 82.8 = 2 092.
        "label": "Qwen3.6-35B-A3B — EXL3 4.0 bpw, MTP (2), chunk 5120 — exllamav3 (tweaked, 12:59Z)",
        "color": "#1f77b4",
        "prefill": [2314, 2566, 2606, 2411, 2178, 2009],
        "decode": [109.4, 110.0, 103.9, 101.3, 93.2, 89.5, 82.8],
    },
    {
        # 2026-11-17 re-run of the tweaked baseline with chunk 6144 —
        # within ~2% of the winner everywhere (128k total 2 079), not a win.
        "label": "Qwen3.6-35B-A3B — EXL3 4.0 bpw, MTP (2), chunk 6144 — exllamav3 (tweaked, 11-17)",
        "color": "#9ecae1",
        "prefill": [2290.7, 2589.1, 2538.6, 2374.7, 2173.8, 1997.7],
        "decode": [108.6, 109.7, 106.8, 102.2, 94.0, 88.1, 81.6],
    },
    {
        # Same tweaked baseline at EXL3 5.0 bpw (Q5), chunk 6144 (run tagged
        # "5120" but 6144 was used), 2026-11-17 17:37Z. 128k total
        # 1 955 + 75.0 ≈ 2 029.6 < 2 092, so the 4.0 bpw run stays the win.
        # See QWEN3.6-35B-A3B.md for the section + Δ vs the 4.0 bpw winner.
        "label": "Qwen3.6-35B-A3B — EXL3 5.0 bpw, MTP (2), chunk 6144 — exllamav3 (tweaked, 17:37Z)",
        "color": "#e00080",
        "prefill": [2054.9, 2452.8, 2450.9, 2328.0, 2127.3, 1954.6],
        "decode": [98.1, 99.3, 93.7, 87.2, 79.3, 79.2, 75.0],
    },
    {
        # Second 11-17 Q5 run — a genuine chunk 5120 (this run's tag matches
        # its behavior, 18:06Z). Decode is clearly better than chunk 6144
        # from 32k to 96k (+4 to +7%), prefill ~4-7% lower. 128k total
        # 1 861 + 74.8 = 1 936 — not a win.
        "label": "Qwen3.6-35B-A3B — EXL3 5.0 bpw, MTP (2), chunk 5120 — exllamav3 (tweaked, 18:06Z)",
        "color": "#bcbd22",
        "prefill": [1986.2, 2273.7, 2337.4, 2205.4, 2001.7, 1861.0],
        "decode": [98.5, 98.2, 90.9, 90.8, 84.5, 81.8, 74.8],
    },
    {
        # 11-17 Q6 tweaked full run, chunk 5120 (tag-genuine, 18:47Z).
        # Prefill ~15-20% below the 4.0 bpw winner; decode ~9-18% below.
        # 128k total 1 722 + 70.7 = 1 793 — not a win.
        "label": "Qwen3.6-35B-A3B — EXL3 6.0 bpw, MTP (2), chunk 5120 — exllamav3 (tweaked, 18:47Z)",
        "color": "#8c564b",
        "prefill": [1856.7, 2000.2, 2108.5, 2026.8, 1844.7, 1722.3],
        "decode": [99.3, 90.4, 87.0, 86.7, 78.6, 78.0, 70.7],
    },
    {
        # 11-17 Q6 tweaked full run, chunk 6144 (button's tag matches here).
        # Prefill is clearly better than 5120 from 16k on (+4.9 to +9.1%),
        # decode notably slower (−4 to −16%), especially on longer runs.
        # 128k total 1 902.7 — still below the 4.0 bpw winner, but best Q6
        # combo for this chart.
        "label": "Qwen3.6-35B-A3B — EXL3 6.0 bpw, MTP (2), chunk 6144 — exllamav3 (tweaked, 19:01Z)",
        "color": "#e6550d",
        "prefill": [1841.3, 2199.7, 2217.4, 2149.3, 1985.6, 1837.0],
        "decode": [85.6, 87.0, 80.0, 75.8, 71.5, 71.2, 65.7],
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