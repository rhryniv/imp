"""Figure: T_N for N=1 (single-cell transmission), for the best
configuration found at each n=1,3,5,7,9, all on one plot.

Note: T_N = 1/(1+|q2|^2 U_{N-1}(kappa_B)^2), and U_0=1 identically, so
T_1 = 1/(1+|q2|^2) = 1/Q -- it does not depend on L at all (the gap
between blocks only adds phase, never attenuates). "Best configuration"
here means: the same p1 (hence Q) recovered for each n in the
realisability run -- L_best only matters for N>=2, but using the
actual recovered p1 (rather than recomputing Qhat_n directly) keeps this
figure tied to the same verified block as the rest of that work.
"""
from __future__ import annotations

import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geometry import setup
from qhat import qhat_n_cheb_coeffs
from spectral import cosine_coeffs, spectral_factor
from spectral import poly_eval

T, U = np.pi / 4, 3 * np.pi / 4
N_VALUES = [1, 3, 5, 7, 9]
COLORS = {1: "#9EC9F0", 3: "#6FB1E8", 5: "#3A86D6", 7: "#1C5FA8", 9: "#0B3D80"}

PASS_TINT = "#DCEEDB"
STOP_TINT = "#FBE8D6"

DPS = 60


def main():
    geo = setup(DPS)
    grid = [mp.mpf(0) + mp.pi * i / 2000 for i in range(2001)]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axvspan(0, float(T), color=PASS_TINT, zorder=0)
    ax.axvspan(float(U), float(np.pi), color=STOP_TINT, zorder=0)

    for n in N_VALUES:
        q, dn = qhat_n_cheb_coeffs(n, geo)
        f = cosine_coeffs(q)
        p1, outside, rho, max_imag = spectral_factor(f, DPS)

        T1 = []
        for vartheta in grid:
            w = mp.e ** (-1j * vartheta)
            Q = abs(poly_eval(p1, w)) ** 2
            T1.append(float(1 / Q))
        ax.plot([float(v) for v in grid], T1, color=COLORS[n], linewidth=1.6, label=f"$n={n}$")

    cosh2mu0 = float(mp.cosh(geo["mu0"]) ** 2)
    ax.axhline(1 / cosh2mu0, color="#999999", linewidth=1.0, linestyle="--")
    ax.text(np.pi, 1 / cosh2mu0, r"  $1/\cosh^2\mu_0$", va="center", ha="left", fontsize=9, color="#666666")

    ax.set_xlim(0, np.pi)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, T, np.pi / 2, U, np.pi])
    ax.set_xticklabels(["0", r"$t=\pi/4$", r"$\pi/2$", r"$u=3\pi/4$", r"$\pi$"])
    ax.set_xlabel(r"$\vartheta$")
    ax.set_ylabel(r"$T_1(\vartheta) = 1/Q(\vartheta)$")
    ax.set_title(r"Single-cell transmission $T_1$ for the best configuration, $n=1,3,5,7,9$")
    ax.text(T / 2, 1.0, r"$I_1$ (pass)", ha="center", va="bottom", fontsize=9, color="#4a7a45")
    ax.text((U + np.pi) / 2, 1.0, r"$I_0$ (stop)", ha="center", va="bottom", fontsize=9, color="#a06a2a")
    ax.legend(loc="center left", frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig("T1_all_n.pdf")
    fig.savefig("T1_all_n.png", dpi=170)
    print("saved T1_all_n.pdf/.png")


if __name__ == "__main__":
    main()
