import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8.5), sharex=True)
ax1.set_xscale("log")

prefill = {
    "Polystrata":          ([4, 16, 32, 64, 96, 126],       [1770, 1963, 1614, 1088, 832, 710]),
    "ExLlamaV3 (recipe)":  ([4, 16, 32, 64, 96, 126],    [670, 1154, 1425, 1344, 1273, 1214]),
    "llama.cpp (no MTP)":  ([4, 16],                       [966, 1032]),
    "llama.cpp (MTP)":     ([4, 16],                       [824, 881]),
}
decode = {
    "Polystrata":          ([0.016, 4, 16, 32, 64, 96, 126], [92.7, 93.1, 84.5, 74.6, 58.5, 49.8, 43.2]),
    "ExLlamaV3 (recipe)":  ([0.016, 4, 16, 32, 64, 96, 126], [75.4, 74.9, 85.4, 67.8, 61.1, 52.7, 50.0]),
    "llama.cpp (no MTP)":  ([0.016, 4, 16],                 [62.7, 61.7, 57.0]),
    "llama.cpp (MTP)":     ([0.016, 4, 16],                 [52.5, 58.1, 50.8]),
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
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", framealpha=0.9)

ax2.set_xscale("log")
ax2.set_xticks([0.016, 4, 16, 32, 64, 96, 126])
ax2.set_xticklabels(["~16", "4k", "16k", "32k", "64k", "96k", "126k"])
ax2.set_xlabel("Prompt / context length")

fig.suptitle("Ornith 1.5 35B-A3B on Radeon 9070 XT — 2026-11-11 runs")
fig.tight_layout()
fig.savefig("benchmarks.png", dpi=200)
print("saved benchmarks.png")