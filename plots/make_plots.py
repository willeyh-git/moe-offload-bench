import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8.5), sharex=True)

# Primary runs: exllamav3 20:53Z (2026-11-11), llama.cpp no-MTP 21:38Z,
# Polystrata 21:13Z; llama.cpp MTP is the earlier 19:56Z run (4k/16k only).
prefill = {
    "Polystrata":          ([4, 16, 32, 64, 96, 128],       [1723, 1938, 1605, 1133, 856, 709]),
    "ExLlamaV3 (recipe)":  ([4, 16, 32, 64, 96, 128],       [1293, 1356, 1344, 1279, 1222, 1168]),
    "llama.cpp (no MTP)":  ([4, 16, 32, 64, 96, 128],       [965, 1033, 1003, 932, 877, 831]),
    "llama.cpp (MTP)":     ([4, 16],                        [824, 881]),
}
decode = {
    "Polystrata":          ([0.02, 4, 16, 32, 64, 96, 128],  [93.9, 91.1, 83.0, 74.3, 58.9, 50.1, 41.1]),
    "ExLlamaV3 (recipe)":  ([0.02, 4, 16, 32, 64, 96, 128],  [94.0, 92.3, 104.5, 82.6, 73.1, 61.7, 57.3]),
    "llama.cpp (no MTP)":  ([0.02, 4, 16, 32, 64, 96, 128],  [62.6, 62.3, 57.0, 52.1, 45.1, 39.2, 35.0]),
    "llama.cpp (MTP)":     ([0.02, 4, 16],                   [52.5, 58.1, 50.8]),
}
colors = {
    "Polystrata":          "#d62728",
    "ExLlamaV3 (recipe)":  "#1f77b4",
    "llama.cpp (no MTP)":  "#2ca02c",
    "llama.cpp (MTP)":     "#9467bd",
}

for ax, data, title in ((ax1, prefill, "Prefill throughput (tok/s)"),
                        (ax2, decode, "Decode throughput (tok/s, median)")):
    for name, (xs, ys) in data.items():
        ax.plot(xs, ys, marker="o", label=name, color=colors[name], lw=1.8, ms=5)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", ncol=2, fontsize=9, framealpha=0.9)

ax2.set_xticks([0.02, 16, 32, 48, 64, 80, 96, 112, 128])
ax2.set_xticklabels(["~16", "16k", "32k", "48k", "64k", "80k", "96k", "112k", "128k"])
ax2.set_xlabel("Prompt / context length")

fig.suptitle("Ornith 1.5 35B-A3B on Radeon 9070 XT — 2026-11-11 runs (llama.cpp MTP: earlier 19:56Z run)")
fig.tight_layout()
fig.savefig("benchmarks.png", dpi=200)
print("saved benchmarks.png")