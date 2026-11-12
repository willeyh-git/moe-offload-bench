import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Qwen3.8-Flash-Next Coder IQ1_M on Polystrata (nov-7 build, --spec 2),
# Radeon 9070 XT / Ryzen 7 5700X3D / 32 GB — llm-bench run
# 20261112T171930Z. Short-context decode (~243 prompt tokens) plotted at
# the left edge; decode min-max spread shown as error bars.
prefill_x = [4, 16, 32, 64, 96, 128]
prefill_y = [1186.1, 1465.5, 1520.0, 1554.3, 1543.6, 1571.6]

decode_x = [0.02, 4, 16, 32, 64, 96, 128]
decode_med = [68.1, 44.6, 46.2, 46.0, 45.1, 45.0, 44.0]
decode_lo = [42.0, 43.2, 44.4, 44.1, 42.8, 45.0, 42.8]
decode_hi = [70.0, 46.2, 49.5, 47.5, 46.2, 45.1, 45.1]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8.5), sharex=True)
ax1.plot(prefill_x, prefill_y, marker="o", color="#d62728", lw=1.8, ms=5,
         label="Polystrata nov-7, --spec 2")
ax1.set_title("Prefill throughput (tok/s)")
ax1.grid(True, alpha=0.25)
ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)

ax2.errorbar(decode_x, decode_med, yerr=[m - lo for m, lo in zip(decode_med, decode_lo)],
             fmt="o", color="#d62728", ecolor="#a05252", elinewidth=1, capsize=3,
             lw=1.8, ms=5, label="Polystrata nov-7, --spec 2 (median)")
ax2.set_title("Decode throughput (tok/s, median, error bars = min-max)")
ax2.grid(True, alpha=0.25)
ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

ax2.set_xticks(decode_x)
ax2.set_xticklabels(["~243", "4k", "16k", "32k", "64k", "96k", "128k"])
ax2.set_xlabel("Prompt / context length")

fig.suptitle("Qwen3.8-Flash-Next Coder IQ1_M on Radeon 9070 XT — Polystrata nov-7 build "
             "(llm-bench 20261112171930Z)")
fig.tight_layout()
fig.savefig("qfn-coder-iq1-m.png", dpi=200)
print("saved qfn-coder-iq1-m.png")