"""Local NLP polish of the (possibly non-tight) lifted-SDP solution.

The lift A ~ a a^T in design_full.py is a genuine *relaxation*: nothing
forces A to actually equal a a^T away from the specific regions where
constraints bite, so the SDP-relaxed a is not always feasible for the true
(nonconvex) constraints (A)/(B)/(C) -- see Remark 5.15 in the paper. Rather
than gamble on rank-1 tightness, we use the SDP solution purely as a warm
start for a local nonlinear polish that enforces the *true* pointwise
constraints (dense grid sampling of theta), which is convex-free but
converges fast from a good starting point and gives a certifiably-correct
answer (checked again with design_full.verify_design on an independent,
finer grid afterwards).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _G_and_grad(a: np.ndarray, theta: np.ndarray, m: np.ndarray):
    cos_mt = np.cos(np.outer(theta, m))
    sin_mt = np.sin(np.outer(theta, m))
    re = cos_mt @ a
    im = sin_mt @ a
    G = re ** 2 + im ** 2
    dG = 2 * re[:, None] * cos_mt + 2 * im[:, None] * sin_mt
    return G, dG


def polish_design(a0, J0, J1, mu0, sigma, n_grid_A=1600, n_grid_local=600, maxiter=500):
    """Refine a0 (from the SDP relaxation) into a locally-optimal, feasible
    solution of the true nonconvex problem via SLSQP on a dense grid.

    Returns (a, u, scipy_result).
    """
    n = len(a0) - 1
    m = np.arange(n + 1)
    theta_A = np.linspace(0.0, np.pi, n_grid_A)
    theta_B = [np.linspace(g, d, n_grid_local) for g, d in J1]
    theta_C = [np.linspace(al, be, n_grid_local) for al, be in J0]

    def objective(x):
        return x[-1]

    def objective_grad(x):
        g = np.zeros(len(x))
        g[-1] = 1.0
        return g

    constraints = []

    def conA(x):
        G, _ = _G_and_grad(x[:-1], theta_A, m)
        return G - 1.0

    def conA_jac(x):
        _, dG = _G_and_grad(x[:-1], theta_A, m)
        J = np.zeros((len(theta_A), len(x)))
        J[:, :-1] = dG
        return J

    constraints.append({"type": "ineq", "fun": conA, "jac": conA_jac})

    for th in theta_B:
        def conB(x, th=th):
            G, _ = _G_and_grad(x[:-1], th, m)
            return x[-1] - G

        def conB_jac(x, th=th):
            _, dG = _G_and_grad(x[:-1], th, m)
            J = np.zeros((len(th), len(x)))
            J[:, :-1] = -dG
            J[:, -1] = 1.0
            return J

        constraints.append({"type": "ineq", "fun": conB, "jac": conB_jac})

    coshmu0 = np.cosh(mu0)
    for th, sig in zip(theta_C, sigma):
        cos_mt = np.cos(np.outer(th, m))

        def conC(x, sig=sig, cos_mt=cos_mt):
            return sig * (cos_mt @ x[:-1]) - coshmu0

        def conC_jac(x, sig=sig, cos_mt=cos_mt):
            J = np.zeros((cos_mt.shape[0], len(x)))
            J[:, :-1] = sig * cos_mt
            return J

        constraints.append({"type": "ineq", "fun": conC, "jac": conC_jac})

    def conD(x):
        return np.array([np.sum(x[:-1]) - 1.0])

    def conD_jac(x):
        J = np.zeros((1, len(x)))
        J[0, :-1] = 1.0
        return J

    constraints.append({"type": "eq", "fun": conD, "jac": conD_jac})

    Gmax_pass = max(np.max(_G_and_grad(a0, th, m)[0]) for th in theta_B) if theta_B else 1.0
    u0 = max(1.0 + 1e-9, Gmax_pass)
    x0 = np.concatenate([a0, [u0]])

    res = minimize(objective, x0, jac=objective_grad, constraints=constraints,
                    method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-14})
    if res.status != 0:
        # SLSQP occasionally stalls on a bad line search; trust-constr is
        # slower but much more robust, use it as a fallback from the same
        # (or SLSQP's improved) starting point.
        x0b = res.x if np.all(np.isfinite(res.x)) else x0
        res2 = minimize(objective, x0b, jac=objective_grad,
                         constraints=[_as_trust_constr(cn) for cn in constraints],
                         method="trust-constr",
                         options={"maxiter": 3000, "gtol": 1e-12, "xtol": 1e-14})
        if res2.fun <= res.fun or res.status != 0:
            res = res2
    a_pol, u_pol = res.x[:-1], res.x[-1]
    return a_pol, float(u_pol), res


def _as_trust_constr(cdict):
    from scipy.optimize import NonlinearConstraint
    fun, jac = cdict["fun"], cdict["jac"]
    if cdict["type"] == "ineq":
        return NonlinearConstraint(fun, 0.0, np.inf, jac=jac)
    return NonlinearConstraint(fun, 0.0, 0.0, jac=jac)
