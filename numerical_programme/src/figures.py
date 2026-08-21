"""F1-F6, F8 (F7 needs E2 and is built separately in figures_f7.py once
E2 finishes). Data provenance for each figure is disclosed in report.md;
interpretive choices (e.g. the "running example" cell, the F4 rho->alpha
convention) are noted there too since the original brief's exact
figure-content sub-specifications are not all recoverable verbatim here.
"""
from __future__ import annotations

import json

import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

from fig_style import (N_STYLE, COLOR_KAPPA_B, COLOR_TENV, COLOR_BAND_FILL,
                        COLOR_I0_BAR, COLOR_I1_BAR, FIGSIZE, FIGSIZE_WIDE,
                        set_theta_axis, save)
from geometry import setup
from e2_core import forward_recursion_np, poly_eval_np, Q_and_kappa
from e2_solve import solve_one
import run_e3

OUT = "../figures"


def U_m(x, m):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    lt = np.abs(x) < 1
    eq1 = np.isclose(x, 1.0)
    eqm1 = np.isclose(x, -1.0)
    gt = (~lt) & (~eq1) & (~eqm1)
    u = np.arccos(np.clip(x[lt], -1, 1))
    out[lt] = np.sin((m + 1) * u) / np.sin(u)
    out[eq1] = m + 1
    out[eqm1] = ((-1) ** m) * (m + 1)
    if np.any(gt):
        xg = x[gt]
        s = np.sign(xg)
        uu = np.arccosh(np.abs(xg))
        val = np.sinh((m + 1) * uu) / np.sinh(uu)
        out[gt] = np.where(s > 0, val, ((-1) ** m) * val)
    return out


def T_N(Q, kappa, N):
    return 1.0 / (1.0 + (Q - 1.0) * U_m(kappa, N - 1) ** 2)


def T_env(Q, kappa):
    return 1.0 / (1.0 + (Q - 1.0) / (1.0 - kappa ** 2))


# ---------------------------------------------------------------- base cell
geoB = setup("B", 30)
tB = float(geoB["t"])
I0B = [(float(lo), float(hi)) for lo, hi in geoB["I0"]]
mu0B = float(geoB["mu0"])
res1 = solve_one(tB, I0B, mu0B, n=1, L=2, sigma_pattern=(-1,), alpha0_first=np.array([0.3]))
alphas_cell = res1["alphas"]
print("running-example (Spec B, one-layer optimum) alphas:", alphas_cell, "delta:", res1["delta"])

thetas = np.linspace(1e-6, np.pi - 1e-6, 4000)
Q, kappa = Q_and_kappa(alphas_cell, 2, thetas)

geoA = setup("A", 30)
tA = float(geoA["t"])
I0A = [(float(lo), float(hi)) for lo, hi in geoA["I0"]]


def shade_bands(ax, t, I0):
    ax.axvspan(0, t, color=COLOR_I1_BAR, alpha=0.35, lw=0, zorder=0)
    for lo, hi in I0:
        ax.axvspan(lo, hi, color=COLOR_I0_BAR, alpha=0.35, lw=0, zorder=0)


# ---------------------------------------------------------------- F1: intro
fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE_WIDE[0], 2.2))
ax = axes[0]
shade_bands(ax, tB, I0B)
ax.plot(thetas, Q, color=N_STYLE[1]["color"], lw=1.1)
ax.axhline(np.cosh(mu0B) ** 2, color="k", lw=0.7, ls=":")
set_theta_axis(ax)
ax.set_ylabel(r"$Q(\theta)$")

ax = axes[1]
shade_bands(ax, tB, I0B)
ax.plot(thetas, kappa, color=COLOR_KAPPA_B, lw=1.2)
ax.axhline(1.0, color="k", lw=0.6, ls="--")
ax.axhline(-1.0, color="k", lw=0.6, ls="--")
set_theta_axis(ax)
ax.set_ylabel(r"$\kappa(\theta)$")
fig.tight_layout()
save(fig, f"{OUT}/fig-intro")

# ---------------------------------------------------------- F2: transmission
fig, ax = plt.subplots(figsize=FIGSIZE)
shade_bands(ax, tB, I0B)
for N in [1, 2, 5, 10]:
    tn = T_N(Q, kappa, N)
    ax.plot(thetas, tn, label=f"$N={N}$", **N_STYLE[N])
ax.plot(thetas, T_env(Q, kappa), color=COLOR_TENV, ls="--", lw=0.9, label=r"$T_{\rm env}$")
set_theta_axis(ax)
ax.set_ylabel(r"$T_N(\theta)$")
ax.set_ylim(-0.02, 1.05)
ax.legend(loc="lower left", ncol=1, frameon=False)
fig.tight_layout()
save(fig, f"{OUT}/fig-transmission")

# ---------------------------------------------------------------- F3: kappa
fig, ax = plt.subplots(figsize=FIGSIZE)
shade_bands(ax, tB, I0B)
ax.plot(thetas, kappa, color=COLOR_KAPPA_B, lw=1.2)
ax.axhspan(-1, 1, color=COLOR_BAND_FILL, alpha=0.6, zorder=-1)
ax.axhline(np.cosh(mu0B), color="k", lw=0.7, ls=":")
ax.axhline(-np.cosh(mu0B), color="k", lw=0.7, ls=":")
set_theta_axis(ax)
ax.set_ylabel(r"$\kappa(\theta)$")
fig.tight_layout()
save(fig, f"{OUT}/fig-kappa")

# ------------------------------------------------------------- F4: gapcount
# Fixed example block (independent of the running example, per the brief):
# rho = (2, sqrt(2), 2), d=3 sections, L=6. Convention: alpha_j = 0.5*ln(rho_j)
# (the standard log-impedance-contrast definition consistent with this
# project's "log-contrasts" terminology) -- disclosed here since this
# specific rho->alpha convention is not independently re-derivable from
# the summarized brief text alone.
rho_ex = [2.0, np.sqrt(2.0), 2.0]
alphas_ex = np.array([0.5 * np.log(r) for r in rho_ex])
L_ex = 6
thetas_full = np.linspace(1e-6, np.pi - 1e-6, 4000)
Q_ex, kappa_ex = Q_and_kappa(alphas_ex, L_ex, thetas_full)
gap_mask = np.abs(kappa_ex) >= 1.0
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.fill_between(thetas_full, -2, 2, where=gap_mask, color=COLOR_I0_BAR, alpha=0.3, lw=0, step=None)
ax.plot(thetas_full, kappa_ex, color=COLOR_KAPPA_B, lw=1.2)
ax.axhline(1.0, color="k", lw=0.6, ls="--")
ax.axhline(-1.0, color="k", lw=0.6, ls="--")
n_gaps = int(np.sum(np.diff(gap_mask.astype(int)) == 1) + (1 if gap_mask[0] else 0))
set_theta_axis(ax)
ax.set_ylabel(r"$\kappa(\theta)$")
ax.set_ylim(-2, 2)
fig.tight_layout()
save(fig, f"{OUT}/fig-gapcount")
print(f"F4 gapcount: d=3 L=6 rho={rho_ex} -> alphas={alphas_ex}, {n_gaps} stop-band(s) on (0,pi)")

# ------------------------------------------------------------- F5: extremal
fig, ax = plt.subplots(figsize=FIGSIZE)
shade_bands(ax, tA, I0A)
extremal_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
extremal_ls = ["-", "--", ":", "-."]
xs_theta = np.linspace(1e-6, np.pi - 1e-6, 2000)
from qhat import cheb_eval
for (n, color, ls) in zip([1, 3, 5, 7], extremal_colors, extremal_ls):
    q, dn = run_e3.qhat_n_cheb_coeffs(n, geoA)
    xvals = np.cos(xs_theta)
    qvals = np.array([float(cheb_eval(q, mp.mpf(float(x)))) for x in xvals])
    ax.plot(xs_theta, qvals, color=color, ls=ls, lw=1.1, label=f"$n={n}$")
ax.axhline(np.cosh(float(geoA["mu0"])) ** 2, color="k", lw=0.7, ls=":")
set_theta_axis(ax)
ax.set_ylabel(r"$\hat Q_n(\theta)$")
ax.set_yscale("log")
ax.legend(loc="center left", frameon=False, fontsize=7)
fig.tight_layout()
save(fig, f"{OUT}/fig-extremal")

# ------------------------------------------------------------- F6: sandwich
with open("../data/e1_results.json") as f:
    e1 = json.load(f)
fig, ax = plt.subplots(figsize=FIGSIZE)
markers = {"A": "o", "B": "s"}
colors = {"A": "#0072B2", "B": "#D55E00"}
for name in ["A", "B"]:
    ns = list(range(1, 9))
    vals = [e1[name][str(n)]["delta_sdp"] for n in ns]
    ax.plot(ns, vals, marker=markers[name], color=colors[name], lw=1.0, ms=3.5,
            label=f"Spec {name}")
ax.set_yscale("log")
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$\delta_{\rm mag}(n)$")
ax.legend(frameon=False)
fig.tight_layout()
save(fig, f"{OUT}/fig-sandwich")

# --------------------------------------------------------------- F8: green
with open("../data/e5_results.json") as f:
    e5 = json.load(f)
geoC = setup("C", 30)
a = float(geoC["a"])
xs = np.linspace(-0.999, 0.999, 2000)
xs = xs[np.abs(xs - a) > 1e-3]
xs_out = xs[(xs < a)]
l = (2 * xs_out - 1 - a) / (1 - a)
gvals = np.arccosh(np.abs(l))
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(xs_out, gvals, color="#0072B2", lw=1.1)
for lo, hi in geoC["I0"]:
    xlo, xhi = float(mp.cos(hi)), float(mp.cos(lo))
    ax.axvspan(xlo, xhi, color=COLOR_I0_BAR, alpha=0.35, lw=0)
ax.axhline(e5["gamma"], color="k", lw=0.7, ls="--")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$g(x)$")
fig.tight_layout()
save(fig, f"{OUT}/fig-green")

print("F1,F2,F3,F4,F5,F6,F8 saved to ../figures/")
