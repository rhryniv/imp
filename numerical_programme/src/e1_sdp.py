"""E1: delta_mag(n) via (a) discretized semi-infinite LP and (b) exact
SDP (Gram/Markov-Lukacs), for any geometry (I_1=[0,t] single interval,
I_0 a list of theta-intervals, possibly disconnected -- Spec C).
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp

from cheb_mk import markov_lukacs_cheb


def theta_to_F0(I0_theta):
    """List of theta-intervals -> list of x-intervals (cos is decreasing,
    so (lo,hi) in theta -> (cos(hi), cos(lo)) in x)."""
    return [(float(np.cos(hi)), float(np.cos(lo))) for lo, hi in I0_theta]


def Q_grid_matrix(thetas, n):
    """Rows: [1, 2cos(theta), 2cos(2theta), ..., 2cos(n theta)] so that
    Q(theta) = row @ f for f=(f_0,...,f_n)."""
    M = np.ones((len(thetas), n + 1))
    for m in range(1, n + 1):
        M[:, m] = 2 * np.cos(m * thetas)
    return M


def solve_lp(n, t, I0_theta, mu0, m_grid=4000, m_refine=20000, solver=cp.CLARABEL):
    """(a) Discretized LP: grid theta on [0,pi] (m_grid pts), refined to
    m_refine on I_1 and each I_0 component. Returns (delta, status)."""
    thetas_global = np.linspace(0.0, np.pi, m_grid)
    thetas_I1 = np.linspace(0.0, t, m_refine)
    thetas_I0 = np.concatenate([np.linspace(lo, hi, m_refine) for lo, hi in I0_theta])

    f = cp.Variable(n + 1)
    delta = cp.Variable(nonneg=True)

    A_glob = Q_grid_matrix(thetas_global, n)
    A_I1 = Q_grid_matrix(thetas_I1, n)
    A_I0 = Q_grid_matrix(thetas_I0, n)
    cosh2mu0 = float(np.cosh(mu0) ** 2)

    constraints = [
        A_glob @ f >= 1.0,
        A_I1 @ f <= 1.0 + delta,
        A_I0 @ f >= cosh2mu0,
        f[0] + 2 * cp.sum(f[1:]) == 1.0,
    ]
    prob = cp.Problem(cp.Minimize(delta), constraints)
    prob.solve(solver=solver)
    return (float(delta.value) if delta.value is not None else None), prob.status


def solve_sdp(n, t, I0_theta, mu0, solver=cp.CLARABEL):
    """(b) Exact SDP: Chebyshev-basis Markov-Lukacs on [-1,1] (A), F_1 (B),
    each F_0 component (C'). Returns (delta, status, gram_vars for reuse)."""
    a = float(np.cos(t))
    F0 = theta_to_F0(I0_theta)
    cosh2mu0 = float(np.cosh(mu0) ** 2)

    # cheb-T coeffs q: q_0=f_0, q_m=2 f_m (Q(theta)=sum q_m T_m(cos theta))
    f = cp.Variable(n + 1)
    delta = cp.Variable(nonneg=True)
    q = cp.hstack([f[0]] + [2 * f[m] for m in range(1, n + 1)])
    e0 = np.zeros(n + 1)
    e0[0] = 1.0

    constraints = [f[0] + 2 * cp.sum(f[1:]) == 1.0]
    gram_vars = []

    consA, gvA, _ = markov_lukacs_cheb(q - e0, n, -1.0, 1.0)
    constraints += consA
    gram_vars += gvA

    consB, gvB, _ = markov_lukacs_cheb((1.0 + delta) * e0 - q, n, a, 1.0)
    constraints += consB
    gram_vars += gvB

    for (fa, fb) in F0:
        consC, gvC, _ = markov_lukacs_cheb(q - cosh2mu0 * e0, n, fa, fb)
        constraints += consC
        gram_vars += gvC

    prob = cp.Problem(cp.Minimize(delta), constraints)
    prob.solve(solver=solver)
    return (float(delta.value) if delta.value is not None else None), prob.status, f.value


if __name__ == "__main__":
    from geometry import setup, delta_n_closed

    for name in ["A", "B"]:
        geo = setup(name, 30)
        t = float(geo["t"])
        I0 = [(float(lo), float(hi)) for lo, hi in geo["I0"]]
        mu0 = float(geo["mu0"])
        print(f"=== Spec {name} ===")
        for n in [1, 3, 5]:
            dlp, slp = solve_lp(n, t, I0, mu0)
            dsdp, ssdp, _ = solve_sdp(n, t, I0, mu0)
            closed = float(delta_n_closed(n, geo))
            print(f"  n={n}  LP delta={dlp:.8e} ({slp})  SDP delta={dsdp:.8e} ({ssdp})  "
                  f"closed={closed:.8e}  SDP reldiff={abs(dsdp-closed)/closed:.2e}")
