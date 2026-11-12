import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Qwen3.6-35B-A3B on ExLlamaV3 (Radeon 9070 XT), 2026-11-12 runs.
# One figure per quant; 3 spec modes each (none / MTP / DFlash2).
# Short-context decode (~90 prompt tokens) plotted at x=0.09.
quants = {
    "q4": {
        "title": "Qwen3.6-35B-A3B — 4.00 bpw (yeasah)",
        "out": "qwen3.6-4.00bpw.png",
        "xs": [0.09, 4, 16, 32],
        "prefill": {
            "none":     [1643.1, 1657.3, 1612.2],
            "MTP":      [1380.3, 1433.5, 1402.4],
            "DFlash2":  [1307.0, 1370.0, 1361.0],
        },
        "decode": {
            "none":     [77.5, 76.8, 74.2, 71.5],
            "MTP":      [102.3, 102.8, 101.0, 98.0],
            "DFlash2":  [113.0, 105.2, 98.4, 92.2],
        },
    },
    "q5": {
        "title": "Qwen3.6-35B-A3B — 5.00 bpw (yeasah)",
        "out": "qwen3.6-5.00bpw.png",
        "xs": [0.09, 4, 16],
        "prefill": {
            "none":     [1218.5, 1228.1],
            "MTP":      [1051.9, 1090.9],
            "DFlash2":  [1060.9, 1093.5],
        },
        "decode": {
            "none":     [71.5, 71.2, 69.0],
            "MTP":      [96.4, 96.5, 89.1],
            "DFlash2":  [79.1, 87.3, 77.8],
        },
    },
    "q6": {
        "title": "Qwen3.6-35B-A3B — 6.00 bpw -hb6 (darkbit1001)",
        "out": "qwen3.6-6.00bpw.png",
        "xs": [0.09, 4, 16],
        "prefill": {
            "none":     [995.9, 1003.3],
            "MTP":      [906.9, 924.0],
            "DFlash2":  [921.1, 921.6],
        },
        "decode": {
            "none":     [68.8, 68.9, 67.2],
            "MTP":      [94.0, 86.3, 82.4],
            "DFlash2":  [97.1, 99.0, 88.5],
        },
    },
}

labels = {"none": "no spec", "MTP": "MTP", "DFlash2": "DFlash2 draft"}
colors = {"none": "#2ca02c", "MTP": "#e00080", "DFlash2": "#1f77b4"}


def xtick_labels(xs):
    return ["~90" if x == 0.09 else f"{x:g}k" for x in xs]


for q in quants.values():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8.5), sharex=True)
    for ax, data, title in ((ax1, q["prefill"], "Prefill throughput (tok/s)"),
                            (ax2, q["decode"], "Decode throughput (tok/s, median)")):
        for line in ["none", "MTP", "DFlash2"]:
            # decode includes the short-context point; prefill starts at 4k
            xs = q["xs"] if ax is ax2 else q["xs"][1:1 + len(data[line])]
            ax.plot(xs, data[line], marker="o", label=labels[line],
                    color=colors[line], lw=1.8, ms=5)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax2.set_xticks(q["xs"])
    ax2.set_xticklabels(xtick_labels(q["xs"]))
    ax2.set_xlabel("Prompt / context length")
    fig.suptitle(f"{q['title']} — ExLlamaV3, Radeon 9070 XT (2026-11-12)")
    fig.tight_layout()
    fig.savefig(q["out"], dpi=200)
    print("saved", q["out"])