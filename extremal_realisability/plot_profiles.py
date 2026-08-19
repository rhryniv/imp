import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXT = open("run_all_output.txt").read()

blocks = re.split(r"=== n=(\d+) ===", TEXT)[1:]
profiles = {}
for i in range(0, len(blocks), 2):
    n = int(blocks[i])
    body = blocks[i + 1]
    m = re.search(r"kappa_min vs L profile: \[(.*?)\]\n", body)
    pairs = re.findall(r"\((\d+), ([0-9.eE+-]+)\)", m.group(1))
    profiles[n] = [(int(L), float(v)) for L, v in pairs]

COLORS = {1: "#9EC9F0", 3: "#6FB1E8", 5: "#3A86D6", 7: "#1C5FA8", 9: "#0B3D80"}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
for n, pts in profiles.items():
    Ls = [L / n for L, v in pts]
    vs = [v for L, v in pts]
    ax.plot(Ls, vs, marker="o", markersize=4, color=COLORS[n], linewidth=1.3, label=f"$n={n}$")

ax.axhline(1.0, color="#a06a2a", linewidth=1.2, linestyle="--")
ax.text(6.05, 1.0, r"$\kappa_{\min}=1$ ($\mu_{\rm eff}=\mu_0$ threshold)", va="bottom", ha="right",
        fontsize=9, color="#a06a2a")
ax.set_xlabel(r"$L/n$")
ax.set_ylabel(r"$\kappa_{\min}(n,L) = \min_{I_0}|\kappa_B|$")
ax.set_title(r"(a) linear scale, all $n$")
ax.set_xlim(2, 6.5)
ax.set_ylim(-0.03, 1.35)
ax.legend(loc="upper right", frameon=False, fontsize=10)

floor = 1e-6
for n in [3, 5, 7, 9]:
    pts = profiles[n]
    Ls = [L / n for L, v in pts]
    vs = [max(v, floor) for L, v in pts]
    ax2.semilogy(Ls, vs, marker="o", markersize=4, color=COLORS[n], linewidth=1.0, label=f"$n={n}$")
ax2.axhline(1.0, color="#a06a2a", linewidth=1.2, linestyle="--")
ax2.set_xlabel(r"$L/n$")
ax2.set_ylabel(r"$\kappa_{\min}(n,L)$ (log scale; floored at $10^{-6}$)")
ax2.set_title(r"(b) log scale, $n\geq 3$ only")
ax2.set_ylim(floor / 2, 2)
ax2.legend(loc="lower right", frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig("kappa_min_vs_L.pdf")
fig.savefig("kappa_min_vs_L.png", dpi=170)
print("saved kappa_min_vs_L.pdf/.png")
