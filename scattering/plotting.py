"""Stage 3 deliverable 3 (spec Sec. 9): "Plots: kappa_B and T_N for the
retained design; underline_delta(n) against delta_achieved(n) across the
scan, with n_min_green marked."

  plot_design            -- kappa_B(theta) and T_N(theta) for a single
                             design, matching the deliverable's literal
                             wording (two panels, not three): NOT a reuse
                             of forward.plot_filter, which also plots
                             |q~_1|^2 -- a useful Stage-1-era diagnostic,
                             but a superset of what this deliverable asks
                             for, so kept separate rather than folded in.
  plot_design_from_record -- convenience wrapper taking a driver.py
                             DegreeRecord directly (the natural way to
                             plot "the retained design").
  plot_bound_vs_achieved  -- underline_delta(n) vs delta_achieved(n)
                             across a degree_scan_stage3 scan.

"n_min_green marked": the spec's own formula (Sec. 5.4) makes n_min_green
a function of a delta TARGET, not of n directly. Originally ambiguous
here (whether "delta" meant each degree's own achieved delta or a fixed
target), resolved by the manuscript revision (driver.py's own module
docstring, Task 11): driver.n_min_green is now called with the fixed
`delta_target`, so it is a single per-instance reference value, constant
across the scan -- both curves drawn below (the flat "curve" across n,
and the vertical marker at the retained record's own value) now
necessarily coincide; both are kept since a flat reference line and a
single vertical marker read differently at a glance, not because the
underlying quantity is still ambiguous.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .forward import a_from_alphas, kappa_B, transmission_TN

Interval = tuple[float, float]


def plot_design(alpha: np.ndarray, I0: Sequence[Interval] = (), I1: Sequence[Interval] = (),
                 N_values: Sequence[int] = (), mu0: float | None = None,
                 theta_range: tuple[float, float] = (1e-3, np.pi - 1e-3),
                 n_grid: int = 4000, savepath: str | None = None):
    """kappa_B(theta) and T_N(theta) (one curve per N in N_values) for a
    single design given by its log-contrasts `alpha`. Shades I0 (stop,
    red) / I1 (pass, green) if given. Returns the matplotlib Figure;
    also saves to savepath if given."""
    import matplotlib.pyplot as plt

    a = a_from_alphas(np.asarray(alpha, dtype=float))
    theta = np.linspace(theta_range[0], theta_range[1], n_grid)
    kap = kappa_B(a, theta)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(theta, kap, color="black")
    if mu0 is not None:
        coshmu0 = float(np.cosh(mu0))
        axes[0].axhline(coshmu0, color="gray", lw=0.7, ls="--")
        axes[0].axhline(-coshmu0, color="gray", lw=0.7, ls="--")
    else:
        axes[0].axhline(1.0, color="gray", lw=0.7, ls="--")
        axes[0].axhline(-1.0, color="gray", lw=0.7, ls="--")
    axes[0].set_ylabel(r"$\kappa_B(\theta)$")

    for N in N_values:
        axes[1].plot(theta, transmission_TN(a, theta, N), label=f"N={N}")
    axes[1].set_ylabel(r"$T_N(\theta)$")
    axes[1].set_xlabel(r"$\theta = 2kh$")
    if N_values:
        axes[1].legend()

    for ax in axes:
        for lo, hi in I0:
            ax.axvspan(lo, hi, color="red", alpha=0.15)
        for lo, hi in I1:
            ax.axvspan(lo, hi, color="green", alpha=0.15)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


def plot_design_from_record(record, I0: Sequence[Interval] = (), I1: Sequence[Interval] = (),
                             N_values: Sequence[int] = (), mu0: float | None = None,
                             theta_range: tuple[float, float] = (1e-3, np.pi - 1e-3),
                             n_grid: int = 4000, savepath: str | None = None):
    """plot_design for a driver.py DegreeRecord directly -- the natural
    way to plot "the retained design" (deliverable 3)."""
    if record.alpha is None:
        raise ValueError(f"record for n={record.n} has no alpha (status={record.status!r})")
    return plot_design(np.asarray(record.alpha, dtype=float), I0=I0, I1=I1, N_values=N_values,
                        mu0=mu0, theta_range=theta_range, n_grid=n_grid, savepath=savepath)


def plot_bound_vs_achieved(records: Sequence, retained=None, savepath: str | None = None):
    """underline_delta(n) against delta_achieved(n) across the scan
    (deliverable 3), log-scaled on y (both quantities routinely span
    several orders of magnitude -- see driver.py's own docstring on the
    underline_delta-delta gap), with n_min_green marked two ways -- see
    module docstring."""
    import matplotlib.pyplot as plt

    recs = sorted(records, key=lambda r: r.n)
    ns = [r.n for r in recs]

    fig, ax = plt.subplots(figsize=(7, 5))

    ub_n = [r.n for r in recs if r.underline_delta is not None and r.underline_delta > 0]
    ub_v = [r.underline_delta for r in recs if r.underline_delta is not None and r.underline_delta > 0]
    if ub_n:
        ax.plot(ub_n, ub_v, "o-", color="tab:blue", label=r"$\underline{\delta}(n)$ (certified lower bound)")

    da_n = [r.n for r in recs if r.delta_achieved is not None and r.delta_achieved > 0]
    da_v = [r.delta_achieved for r in recs if r.delta_achieved is not None and r.delta_achieved > 0]
    if da_n:
        ax.plot(da_n, da_v, "s-", color="tab:orange", label=r"$\delta(n)$ (achieved, certified)")

    ng_n = [r.n for r in recs if r.n_min_green is not None]
    ng_v = [r.n_min_green for r in recs if r.n_min_green is not None]
    if ng_n:
        ax.plot(ng_n, ng_v, "^--", color="tab:green", alpha=0.6,
                 label=r"$n_{\min}^{\rm green}(\delta(n))$")

    if retained is not None and retained.n_min_green is not None:
        ax.axvline(retained.n_min_green, color="tab:green", lw=1.2, ls=":",
                    label=r"$n_{\min}^{\rm green}$ at retained $n$")
    if retained is not None:
        ax.axvline(retained.n, color="black", lw=1.0, ls="-", alpha=0.4, label="retained $n$")

    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\delta$")
    ax.set_xticks(ns)
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig
