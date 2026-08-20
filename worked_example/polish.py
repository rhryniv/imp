"""Polish a Task C candidate that the coarse (150-pt) search grid accepted
but fine-grid verification flags as marginally infeasible: re-solve with
a finer grid, seeded from the candidate itself, to remove the narrow gap
the coarse grid missed."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from task_c_core import Q_and_kappa, full_alphas
from task_c_solve import verify_fine


def polish(t, u, mu0, n, L, sigma, alphas0, grid_n=1000):
    vartheta_I1 = np.linspace(0.0, t, grid_n)
    vartheta_I0 = np.linspace(u, np.pi, grid_n)
    cosh_mu0 = np.cosh(mu0)

    def constraints(x):
        alpha_rest, delta_var = x[:n], x[n]
        alphas = full_alphas(alpha_rest)
        Q1, kap1 = Q_and_kappa(alphas, L, vartheta_I1)
        _, kap0 = Q_and_kappa(alphas, L, vartheta_I0)
        c_delta = delta_var - (Q1 - 1.0)
        c_C = sigma * kap0 - cosh_mu0
        c_E_hi = 1.0 - kap1
        c_E_lo = 1.0 + kap1
        return np.concatenate([c_delta, c_C, c_E_hi, c_E_lo])

    alpha_rest0 = np.array(alphas0)[1:]
    Q1, _ = Q_and_kappa(np.array(alphas0), L, vartheta_I1)
    x0 = np.concatenate([alpha_rest0, [max(np.max(Q1 - 1.0), 1e-12)]])
    bounds = [(-5.0, 5.0)] * n + [(0.0, None)]

    res = minimize(lambda x: x[n], x0, method="SLSQP",
                    constraints=[{"type": "ineq", "fun": constraints}],
                    bounds=bounds, options={"maxiter": 300, "ftol": 1e-14})

    alphas = full_alphas(res.x[:n])
    v = verify_fine(t, u, mu0, L, sigma, alphas, grid_n=5000)
    return {"alphas": alphas, "delta": res.x[n], "verify": v, "success": res.success}
