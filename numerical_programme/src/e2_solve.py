"""E2: the NLP solve for one (spec, n, L, sigma_pattern). sigma_pattern
is a tuple of +-1, one per I_0 component. Grids >=2000 pts/interval per
the brief. (A),(D) are asserted (e2_core.assert_sanity), never imposed.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from e2_core import Q_and_kappa, full_alphas, assert_sanity

GRID_N = 2000
FEAS_TOL = 1e-7


def solve_one(t, I0, mu0, n, L, sigma_pattern, alpha0_first):
    theta_I1 = np.linspace(0.0, t, GRID_N)
    theta_I0_list = [np.linspace(lo, hi, GRID_N) for lo, hi in I0]
    cosh_mu0 = np.cosh(mu0)

    def unpack(x):
        return x[:n], x[n]

    def objective(x):
        return x[n]

    def constraints(x):
        alpha_first, delta_var = unpack(x)
        alphas = full_alphas(alpha_first)
        Q1, kap1 = Q_and_kappa(alphas, L, theta_I1)
        c_delta = delta_var - (Q1 - 1.0)
        c_C = []
        for sigma, theta_I0 in zip(sigma_pattern, theta_I0_list):
            _, kap0 = Q_and_kappa(alphas, L, theta_I0)
            c_C.append(sigma * kap0 - cosh_mu0)
        c_E_hi = 1.0 - kap1
        c_E_lo = 1.0 + kap1
        return np.concatenate([c_delta] + c_C + [c_E_hi, c_E_lo])

    x0 = np.concatenate([alpha0_first, [1.0]])
    bounds = [(-5.0, 5.0)] * n + [(0.0, None)]

    res = minimize(objective, x0, method="SLSQP",
                    constraints=[{"type": "ineq", "fun": constraints}],
                    bounds=bounds, options={"maxiter": 150, "ftol": 1e-12})

    alpha_first, delta_var = unpack(res.x)
    alphas = full_alphas(alpha_first)
    cons_val = constraints(res.x)
    feasible = np.all(cons_val >= -FEAS_TOL)
    return {"success": res.success, "delta": delta_var, "alphas": alphas,
            "feasible": feasible, "min_constraint": float(cons_val.min()), "message": res.message}


POLISH_GRID_N = 300  # coarser than GRID_N: keeps the finite-diff Jacobian of
# trust-constr's vector-valued constraint (O(POLISH_GRID_N) rows) tractable.
# The mandated 10x-finer re-verification (verify_fine, hazard #5) is always
# run afterwards at GRID_N*10, so this coarsening only affects the polish
# step's internal grid, never the reported feasibility.


def polish_trust_constr(t, I0, mu0, n, L, sigma_pattern, alphas0, delta0):
    from scipy.optimize import NonlinearConstraint

    theta_I1 = np.linspace(0.0, t, POLISH_GRID_N)
    theta_I0_list = [np.linspace(lo, hi, POLISH_GRID_N) for lo, hi in I0]
    cosh_mu0 = np.cosh(mu0)

    def unpack(x):
        return x[:n], x[n]

    def objective(x):
        return x[n]

    def constraints(x):
        alpha_first, delta_var = unpack(x)
        alphas = full_alphas(alpha_first)
        Q1, kap1 = Q_and_kappa(alphas, L, theta_I1)
        c_delta = delta_var - (Q1 - 1.0)
        c_C = []
        for sigma, theta_I0 in zip(sigma_pattern, theta_I0_list):
            _, kap0 = Q_and_kappa(alphas, L, theta_I0)
            c_C.append(sigma * kap0 - cosh_mu0)
        c_E_hi = 1.0 - kap1
        c_E_lo = 1.0 + kap1
        return np.concatenate([c_delta] + c_C + [c_E_hi, c_E_lo])

    nlc = NonlinearConstraint(constraints, 0.0, np.inf)
    x0 = np.concatenate([alphas0[:n], [delta0]])
    try:
        res = minimize(objective, x0, method="trust-constr", constraints=[nlc],
                        options={"maxiter": 60, "gtol": 1e-10, "xtol": 1e-12})
        alpha_first, delta_var = unpack(res.x)
        success = res.success
    except Exception:
        alpha_first, delta_var = unpack(x0)
        success = False
    alphas = full_alphas(alpha_first)
    return {"delta": delta_var, "alphas": alphas, "success": success}


def verify_fine(t, I0, mu0, sigma_pattern, L, alphas, grid_n=20000, tol=1e-6):
    cosh_mu0 = np.cosh(mu0)
    assert_sanity(alphas)
    theta_I1 = np.linspace(0.0, t, grid_n)
    Q1, kap1 = Q_and_kappa(alphas, L, theta_I1)
    delta = np.max(Q1 - 1.0)
    C_slacks = []
    for sigma, (lo, hi) in zip(sigma_pattern, I0):
        theta_I0 = np.linspace(lo, hi, grid_n)
        _, kap0 = Q_and_kappa(alphas, L, theta_I0)
        C_slacks.append(np.min(sigma * kap0 - cosh_mu0))
    C_ok = all(s >= -tol for s in C_slacks)
    E_ok = bool(np.all(np.abs(kap1) <= 1.0 + tol))
    return {"delta": float(delta), "C_ok": bool(C_ok), "E_ok": E_ok,
            "min_C_slack": float(min(C_slacks)), "min_E_slack": float(np.min(1.0 - np.abs(kap1)))}
