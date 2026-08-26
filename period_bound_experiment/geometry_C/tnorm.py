"""T_N via the Chebyshev recurrence for U_{N-1} -- NOT the closed-form
division formulas (sin/sinh-based) used in earlier experiments, per this
spec's explicit instruction. U_0(x)=1, U_1(x)=2x, U_m(x)=2x U_{m-1}(x) -
U_{m-2}(x); vectorised over an array of kappa values.
"""
from __future__ import annotations

import numpy as np


def U_recurrence(x, m):
    """U_m(x) via the three-term recurrence, x an array, m>=0 integer."""
    x = np.asarray(x, dtype=float)
    if m == 0:
        return np.ones_like(x)
    U_prev = np.ones_like(x)       # U_0
    U_curr = 2.0 * x                # U_1
    for _ in range(2, m + 1):
        U_prev, U_curr = U_curr, 2.0 * x * U_curr - U_prev
    return U_curr


def T_N(Q, kappa, N):
    U = U_recurrence(kappa, N - 1)
    return 1.0 / (1.0 + (Q - 1.0) * U ** 2)


def eps0_direct(Q_I0, kappa_I0, N):
    return float(np.max(T_N(Q_I0, kappa_I0, N)))


def eps1_direct(Q_I1, kappa_I1, N):
    return float(1.0 - np.min(T_N(Q_I1, kappa_I1, N)))


def eps0_formula(beta, mu0, N):
    """[1 + beta*sinh^2(N*mu0)/sinh^2(mu0)]^-1, beta at the active (C) point."""
    denom = 1.0 + beta * (np.sinh(N * mu0) ** 2) / (np.sinh(mu0) ** 2)
    return float(1.0 / denom)
