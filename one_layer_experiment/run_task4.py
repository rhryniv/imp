"""Task 4 -- feasibility region: verify Claim 2 directly on a grid of
(s, mu_0, delta), checking that (B),(C),(E) all hold (numerically, via
the matrix-derived kappa_B/Q, cross-checked in Task 1 to match the
closed forms) exactly when s_-(mu_0) <= s <= min(s_+(delta), cot^2(t/2)).
Baseline t=pi/4, u=3pi/4 (not restated in the brief; using the Task 2
default, flagged in the review).
"""
from __future__ import annotations

import numpy as np

from transfer_matrix_vec import kappa_B_vec
from closed_forms import s_minus, s_plus, cot2_half, delta_min, s_minus_mp, delta_min_mp
import mpmath as mp

T, U = np.pi / 4, 3 * np.pi / 4
L = 2
N_GRID_INTERVAL = 800  # points per I0/I1 for the "for all vartheta" check
TOL = 1e-9


def constraints_hold(s, mu0, delta):
    alpha = np.arcsinh(np.sqrt(s))
    grid_I1 = np.linspace(0.0, T, N_GRID_INTERVAL)
    grid_I0 = np.linspace(U, np.pi, N_GRID_INTERVAL)

    kap1, q1_1, q2_1 = kappa_B_vec(alpha, grid_I1 / 2, L)
    Q1 = np.abs(q1_1) ** 2
    B_ok = np.all(Q1 <= 1 + delta + TOL)
    E_ok = np.all((kap1 >= -1 - TOL) & (kap1 <= 1 + TOL))

    kap0, q1_0, q2_0 = kappa_B_vec(alpha, grid_I0 / 2, L)
    C_ok = np.all(-kap0 >= np.cosh(mu0) - TOL)  # sigma_1 = -1

    return B_ok, C_ok, E_ok


def run():
    s_grid = np.linspace(0.05, 3.0, 20)
    mu0_grid = np.linspace(0.1, 2.0, 12)
    delta_grid = np.geomspace(1e-3, 2.0, 12)

    disagreements = []
    n_checked = 0
    for mu0 in mu0_grid:
        sm = s_minus(mu0, U)
        for delta in delta_grid:
            sp = s_plus(delta, T)
            cot2 = cot2_half(T)
            upper = min(sp, cot2)
            for s in s_grid:
                n_checked += 1
                closed_ok = (sm - 1e-9 <= s <= upper + 1e-9)
                B_ok, C_ok, E_ok = constraints_hold(s, mu0, delta)
                numeric_ok = B_ok and C_ok and E_ok
                if closed_ok != numeric_ok:
                    disagreements.append((s, mu0, delta, closed_ok, numeric_ok, B_ok, C_ok, E_ok))

    print(f"checked {n_checked} (s, mu0, delta) points")
    print(f"disagreements: {len(disagreements)}")
    for d in disagreements[:20]:
        print(" ", d)
    return disagreements


def confirm_extras():
    print("\n=== extra confirmations ===")
    cot2_t = cot2_half(T)
    print(f"cot^2(pi/8) = {cot2_t:.4f}  (brief states 5.8284...)")

    print("delta_min(mu0) at several mu0 (baseline t,u), high precision:")
    mp.mp.dps = 30
    for mu0 in [0.25, 0.5, 1, 2]:
        dm = delta_min_mp(mp.mpf(mu0), mp.pi / 4, 3 * mp.pi / 4)
        print(f"  mu0={mu0}: delta_min = {mp.nstr(dm, 12)}")

    print("s_-(mu0) > 0 always:")
    for mu0 in [0.01, 0.1, 0.5, 1, 2, 5]:
        sm = s_minus(mu0, U)
        print(f"  mu0={mu0}: s_- = {sm:.6f}  (>0: {sm > 0})")


if __name__ == "__main__":
    run()
    confirm_extras()
