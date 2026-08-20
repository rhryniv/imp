"""Task C: the NLP solve for one (geometry, n, L, sigma) combination."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from task_c_core import Q_and_kappa, full_alphas

GRID_N = 150
FEAS_TOL = 1e-7


def solve_one(t, u, mu0, n, L, sigma, alpha0_rest):
    vartheta_I1 = np.linspace(0.0, t, GRID_N)
    vartheta_I0 = np.linspace(u, np.pi, GRID_N)
    cosh_mu0 = np.cosh(mu0)

    def unpack(x):
        return x[:n], x[n]

    def objective(x):
        return x[n]

    def constraints(x):
        alpha_rest, delta_var = unpack(x)
        alphas = full_alphas(alpha_rest)
        Q1, kap1 = Q_and_kappa(alphas, L, vartheta_I1)
        _, kap0 = Q_and_kappa(alphas, L, vartheta_I0)
        c_delta = delta_var - (Q1 - 1.0)
        c_C = sigma * kap0 - cosh_mu0
        c_E_hi = 1.0 - kap1
        c_E_lo = 1.0 + kap1
        return np.concatenate([c_delta, c_C, c_E_hi, c_E_lo])

    x0 = np.concatenate([alpha0_rest, [1.0]])  # generous initial delta_var
    bounds = [(-5.0, 5.0)] * n + [(0.0, None)]

    res = minimize(objective, x0, method="SLSQP",
                    constraints=[{"type": "ineq", "fun": constraints}],
                    bounds=bounds, options={"maxiter": 150, "ftol": 1e-12})

    alpha_rest, delta_var = unpack(res.x)
    alphas = full_alphas(alpha_rest)
    cons_val = constraints(res.x)
    feasible = np.all(cons_val >= -FEAS_TOL)
    return {"success": res.success, "delta": delta_var, "alphas": alphas,
            "feasible": feasible, "min_constraint": cons_val.min(), "message": res.message}


def verify_fine(t, u, mu0, L, sigma, alphas, grid_n=3000, tol=1e-6):
    vartheta_I1 = np.linspace(0.0, t, grid_n)
    vartheta_I0 = np.linspace(u, np.pi, grid_n)
    Q1, kap1 = Q_and_kappa(alphas, L, vartheta_I1)
    _, kap0 = Q_and_kappa(alphas, L, vartheta_I0)
    delta = np.max(Q1 - 1.0)
    C_ok = np.all(sigma * kap0 - np.cosh(mu0) >= -tol)
    E_ok = np.all(np.abs(kap1) <= 1.0 + tol)
    return {"delta": delta, "C_ok": bool(C_ok), "E_ok": bool(E_ok),
            "min_C_slack": float(np.min(sigma * kap0 - np.cosh(mu0))),
            "min_E_slack": float(np.min(1.0 - np.abs(kap1)))}
