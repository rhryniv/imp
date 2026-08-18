import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp

mp.mp.dps = 50

d = json.load(open("taskB_results.json"))
gammas = {"baseline": 3.0571418389619964, "narrower": 2.2923930300783895,
          "wider": 3.983297385668894, "mu0=0.5": 3.0571418389619964, "mu0=2": 3.0571418389619964}
# recompute exactly to avoid a stale hardcode
import numpy as np
from variants import Geometry
CASES = {
    "baseline": Geometry(t=np.pi / 4, u=3 * np.pi / 4, mu0=1.0, t_frac=(1, 4), u_frac=(3, 4)),
    "narrower": Geometry(t=np.pi / 3, u=2 * np.pi / 3, mu0=1.0, t_frac=(1, 3), u_frac=(2, 3)),
    "wider": Geometry(t=np.pi / 6, u=5 * np.pi / 6, mu0=1.0, t_frac=(1, 6), u_frac=(5, 6)),
    "mu0=0.5": Geometry(t=np.pi / 4, u=3 * np.pi / 4, mu0=0.5, t_frac=(1, 4), u_frac=(3, 4)),
    "mu0=2": Geometry(t=np.pi / 4, u=3 * np.pi / 4, mu0=2.0, t_frac=(1, 4), u_frac=(3, 4)),
}
gammas = {name: float(geo.gamma) for name, geo in CASES.items()}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
markers = {"baseline": "o", "narrower": "s", "wider": "^", "mu0=0.5": "D", "mu0=2": "v"}
colors = {"baseline": "tab:blue", "narrower": "tab:orange", "wider": "tab:green",
          "mu0=0.5": "tab:red", "mu0=2": "tab:purple"}

for name, res in d.items():
    gamma = gammas[name]
    ns_odd, ys_odd, ns_even, ys_even = [], [], [], []
    for n_str, r in sorted(res.items(), key=lambda kv: int(kv[0])):
        if not r.get("ok"):
            continue
        n = int(n_str)
        delta = mp.mpf(r["delta"])
        y = float(mp.log(delta) + gamma * n)
        if n % 2 == 1:
            ns_odd.append(n)
            ys_odd.append(y)
        else:
            ns_even.append(n)
            ys_even.append(y)
    axes[0].plot(ns_odd, ys_odd, marker=markers[name], color=colors[name], linestyle="-",
                 label=f"{name} (odd n)")
    axes[0].plot(ns_even, ys_even, marker=markers[name], color=colors[name], linestyle="--",
                 alpha=0.5, label=f"{name} (even n)")

axes[0].set_xlabel("degree n")
axes[0].set_ylabel(r"$\log \delta_{mag}(n) + \gamma n$")
axes[0].set_title("Rate-attainment test: flat = rate attained")
axes[0].legend(fontsize=7, ncol=1)
axes[0].grid(True, alpha=0.3)

# right panel: baseline only, delta vs analytic bound, log scale
res = d["baseline"]
gamma = gammas["baseline"]
ns = sorted(int(k) for k, v in res.items() if v.get("ok"))
deltas = [float(mp.mpf(res[str(n)]["delta"])) for n in ns]
beta1 = float(CASES["baseline"].beta1)
analytic = [beta1 * float(mp.e ** (-gamma * n)) for n in ns]
axes[1].semilogy(ns, deltas, "o-", label=r"$\delta_{mag}(n)$ (this work, 50-digit)")
axes[1].semilogy(ns, analytic, "k:", linewidth=2, label=r"$\sinh^2\mu_0\, e^{-\gamma n}$")
axes[1].set_xlabel("degree n")
axes[1].set_ylabel(r"$\delta$ (log scale)")
axes[1].set_title("Baseline: certified value vs analytic bound")
axes[1].legend(fontsize=8)
axes[1].grid(True, which="both", alpha=0.3)

fig.tight_layout()
fig.savefig("taskAB_plot.png", dpi=150)
print("saved taskAB_plot.png")
