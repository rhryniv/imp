import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open("results.json"))
ns = list(range(1, 9))
gamma = 3.057142
analytic = [1.3811 * np.exp(-gamma * n) for n in ns]

fig, ax = plt.subplots(figsize=(7, 5))
colors = {"I": "tab:blue", "II": "tab:orange"}
for variant in ("I", "II"):
    row_p = [x for x in r if x["variant"] == variant and x["solver"] == "CLARABEL"]
    row_p.sort(key=lambda x: x["n"])
    dp = [x["delta_primal"] for x in row_p]
    du = [x["underline_delta"] for x in row_p]
    ax.semilogy(ns, dp, "o-", color=colors[variant], label=f"$\\delta^{{primal}}$ variant {variant} (CLARABEL)")
    ax.semilogy(ns, np.abs(du), "s--", color=colors[variant], alpha=0.6,
                label=f"$\\underline{{\\delta}}$ variant {variant} (CLARABEL)")

ax.semilogy(ns, analytic, "k:", linewidth=2, label=r"$1.3811\,e^{-3.06n}$ (analytic bound)")
ax.set_xlabel("degree n")
ax.set_ylabel(r"$\delta$ (log scale)")
ax.set_title("SDP certificate test: primal/dual bounds vs analytic bound")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("cert_plot.png", dpi=150)
print("saved cert_plot.png")
