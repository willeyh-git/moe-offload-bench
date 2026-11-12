import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Qwen3.6-35B-A3B on ExLlamaV3 (Radeon 9070 XT), 2026-11-12 runs.
# One figure per quant; 3 spec modes each (none / MTP / DFlash2), plus an
# aside "MTP (2 drafts)" run for 4.00 bpw (11:20Z, draft tokens 2,
# cpu_moe_split_experts 95, full range) drawn dashed.
# Short-context decode (~90 prompt tokens) plotted at x=0.09.
quants = {
    "q4": {
        "title": "Qwen3.6-35B-A3B — 4.00 bpw (yeasah)",
        "out": "qwen3.6-4.00bpw.png",
        "xs": [0.09, 4, 16, 32, 64, 96, 126],
        "labels": {"MTP": "MTP (5 drafts)", "DFlash2": "DFlash2 (5 drafts)"},
        "prefill": {
            "none":      [1643.1, 1657.3, 1612.2],
            "MTP":       [1380.3, 1433.5, 1402.4],
            "DFlash2":   [1307.0, 1370.0, 1361.0],
            "MTP (2 drafts)": [1419.8, 1473.0, 1438.6, 1351.7, 1273.3, 1211.2],
            "DFlash2 (2 drafts)": [1463.0, 1534.3, 1492.1, 1398.3, 1312.4, 1245.6],
        },
        "decode": {
            "none":      [77.5, 76.8, 74.2, 71.5],
            "MTP":       [102.3, 102.8, 101.0, 98.0],
            "DFlash2":   [113.0, 105.2, 98.4, 92.2],
            "MTP (2 drafts)": [112.8, 113.2, 105.0, 100.7, 93.6, 91.7, 83.3],
            "DFlash2 (2 drafts)": [105.4, 107.0, 98.9, 93.0, 86.3, 80.1, 71.6],
        },
    },
    "q5": {
        "title": "Qwen3.6-35B-A3B — 5.00 bpw (yeasah)",
        "out": "qwen3.6-5.00bpw.png",
        "xs": [0.09, 4, 16, 32, 64, 96, 126],
        "labels": {"MTP": "MTP (4 drafts)", "DFlash2": "DFlash2 (5 drafts)"},
        "prefill": {
            "none":     [1218.5, 1228.1],
            "MTP":      [1051.9, 1090.9],
            "DFlash2":  [1060.9, 1093.5],
            "DFlash2 (2 drafts)": [1128.0, 1155.0, 1129.9, 1074.1, 1011.3, 978.2],
        },
        "decode": {
            "none":     [71.5, 71.2, 69.0],
            "MTP":      [96.4, 96.5, 89.1],
            "DFlash2":  [79.1, 87.3, 77.8],
            "DFlash2 (2 drafts)": [98.7, 99.2, 94.0, 86.6, 80.5, 76.1, 67.2],
        },
    },
    "q6": {
        "title": "Qwen3.6-35B-A3B — 6.00 bpw -hb6 (darkbit1001)",
        "out": "qwen3.6-6.00bpw.png",
        "xs": [0.09, 4, 16],
        "labels": {"MTP": "MTP (4 drafts)", "DFlash2": "DFlash2 (5 drafts)"},
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

# draw order, color, line style
order = ["none", "MTP", "DFlash2", "MTP (2 drafts)", "DFlash2 (2 drafts)"]
colors = {"none": "#2ca02c", "MTP": "#e00080", "DFlash2": "#1f77b4",
          "MTP (2 drafts)": "#c0392b", "DFlash2 (2 drafts)": "#ff7f0e"}
styles = {"none": "-", "MTP": "-", "DFlash2": "-",
          "MTP (2 drafts)": "--", "DFlash2 (2 drafts)": "--"}


def xtick_labels(xs):
    return ["~90" if x == 0.09 else f"{x:g}k" for x in xs]


for q in quants.values():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8.5), sharex=True)
    for ax, data, title in ((ax1, q["prefill"], "Prefill throughput (tok/s)"),
                            (ax2, q["decode"], "Decode throughput (tok/s, median)")):
        for line in order:
            if line not in data:
                continue
            # decode includes the short-context point; prefill starts at 4k
            xs = q["xs"][:len(data[line])] if ax is ax2 else q["xs"][1:1 + len(data[line])]
            # aside (2 drafts) lines: dashed + distinct marker + distinct color
            marker = "s" if "(2 drafts)" in line else "o"
            ax.plot(xs, data[line], marker=marker, label=q["labels"].get(line, line),
                    color=colors[line], ls=styles[line], lw=1.8, ms=5)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        for h, t in zip(*ax.get_legend_handles_labels()):
            print(q["out"], t, h.get_color(), h.get_linestyle())
    ax2.set_xticks(q["xs"])
    ax2.set_xticklabels(xtick_labels(q["xs"]))
    ax2.set_xlabel("Prompt / context length")
    fig.suptitle(f"{q['title']} — ExLlamaV3, Radeon 9070 XT (2026-11-12)")
    fig.tight_layout()
    fig.savefig(q["out"], dpi=200)
    print("saved", q["out"])