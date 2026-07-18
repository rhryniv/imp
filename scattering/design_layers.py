"""Design the single-block filter by optimizing directly over the physical
Schur/layer parameters gamma_j = tanh(alpha_j) in (-1,1), j=0,...,n,
instead of over the coefficient vector a = q~_1 directly.

Rationale (per discussion): the freedom in Remark 4.9 ("any choice of the
spectral factor is admissible") is about q_2 only -- it doesn't touch
kappa_B or T_N. Those are fixed once q_1 is fixed. The issue found with
design_full.py + polish.py was that the *optimizer* was not restricted to
searching among realizable q_1's at all: a generic real vector a
satisfying only the magnitude constraint (A) need not be zero-free (e.g.
a=[3,2] and its reversal a=[2,3] give the same |q~_1| but only one is
zero-free), so the solver could wander onto a branch with no physical
layer sequence behind it.

Parametrizing by (gamma_0,...,gamma_n) in (-1,1)^{n+1} and building q~_1
via the forward recursion (eq. 4.5-4.6, forward_reconstruct) sidesteps
this completely: every point in that box is realizable by construction
(Prop 4.4), and constraint (A) |q~_1| >= 1 holds identically (it's exactly
the SU(1,1)/Cayley-Klein identity |p1|^2-|p2|^2=1), so it doesn't need to
be imposed at all -- only (B) pass-band flatness, (C) stop-band depth, and
(D) sum(alpha_j)=0 remain.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from .layer_stripping import forward_reconstruct
from .transmission import kappa_B
from .spectral_factor import evaluate_G_from_a
from .design import Interval


@dataclass
class LayerDesignResult:
    n: int
    mu0: float
    status: str
    sigma: tuple | None = None
    gamma: np.ndarray | None = None
    alphas: np.ndarray | None = None
    a: np.ndarray | None = None
    impedances: np.ndarray | None = None
    u: float | None = None
    delta1: float | None = None
    verified: bool | None = None


def _free_to_alphas(gamma_free: np.ndarray) -> np.ndarray:
    """gamma_free = (gamma_0,...,gamma_{n-1}) are free; alpha_n is *solved
    for* so that sum(alphas) = 0 (constraint D) automatically -- this
    removes D as an explicit equality constraint (which made SLSQP struggle
    badly, see the design_layers.py docstring / commit history)."""
    alphas_free = np.arctanh(gamma_free)
    alpha_n = -np.sum(alphas_free)
    return np.concatenate([alphas_free, [alpha_n]])


def _a_from_gamma_free(gamma_free: np.ndarray) -> np.ndarray:
    alphas = _free_to_alphas(gamma_free)
    p1, _ = forward_reconstruct(alphas)
    return p1[::-1]


def _solve_for_sigma(n, J0, J1, mu0, sigma, gamma_bound, n_grid_B, n_grid_C, n_restarts, rng, gamma_free_init=None):
    theta_B = [np.linspace(g, d, n_grid_B) for g, d in J1]
    theta_C = [np.linspace(al, be, n_grid_C) for al, be in J0]

    def objective(x):
        return x[-1]

    def objective_grad(x):
        g = np.zeros(len(x))
        g[-1] = 1.0
        return g

    constraints = []
    for th in theta_B:
        def fun(x, th=th):
            a = _a_from_gamma_free(x[:-1])
            return x[-1] - evaluate_G_from_a(a, th)
        constraints.append({"type": "ineq", "fun": fun})

    coshmu0 = np.cosh(mu0)
    for th, sig in zip(theta_C, sigma):
        def fun(x, th=th, sig=sig):
            a = _a_from_gamma_free(x[:-1])
            return sig * kappa_B(a, th) - coshmu0
        constraints.append({"type": "ineq", "fun": fun})

    # constraint (D), sum(alphas) = 0, is enforced *by construction* via
    # _free_to_alphas (alpha_n is solved for), so only n free gammas remain.
    bounds = [(-gamma_bound, gamma_bound)] * n + [(1.0, None)]

    base = gamma_free_init if gamma_free_init is not None else np.zeros(n)
    base = np.clip(base, -gamma_bound, gamma_bound)

    from .design_full import verify_design

    # SLSQP frequently reports a nonzero exit status ("positive directional
    # derivative", etc.) right next to a genuinely good, feasible point --
    # a known quirk near active/nonsmooth constraints. So instead of
    # trusting res.status, verify each candidate directly on a fine grid
    # and keep the best one that actually is feasible, regardless of the
    # solver's own exit flag.
    best_verified = None
    best_any = None
    for attempt in range(n_restarts):
        noise = 0.15 * rng.standard_normal(n) if attempt > 0 else 0.0
        gamma0 = np.clip(base + noise, -gamma_bound, gamma_bound)
        x0 = np.concatenate([gamma0, [2.0]])
        res = minimize(objective, x0, jac=objective_grad, constraints=constraints, bounds=bounds,
                        method="SLSQP", options={"maxiter": 400, "ftol": 1e-14})
        if not np.all(np.isfinite(res.x)):
            continue
        a_try = _a_from_gamma_free(res.x[:-1])
        u_try = float(res.x[-1])
        if verify_design(a_try, J0, J1, mu0, sigma, u_try, tol=1e-6):
            if best_verified is None or u_try < best_verified.fun:
                best_verified = res
        if best_any is None or res.fun < best_any.fun:
            best_any = res

    if best_verified is None and best_any is None:
        return LayerDesignResult(n=n, mu0=mu0, status="no_restart_converged", sigma=sigma)

    best = best_verified if best_verified is not None else best_any
    gamma_free = best.x[:-1]
    u = float(best.x[-1])
    alphas = _free_to_alphas(gamma_free)
    gamma = np.tanh(alphas)
    a = _a_from_gamma_free(gamma_free)
    impedances = np.empty(n + 2)
    impedances[0] = 1.0
    for j, alpha_j in enumerate(alphas):
        impedances[j + 1] = impedances[j] * np.exp(alpha_j)

    verified = verify_design(a, J0, J1, mu0, sigma, u, tol=1e-6)

    return LayerDesignResult(
        n=n, mu0=mu0, status="optimal", sigma=sigma, gamma=gamma, alphas=alphas, a=a,
        impedances=impedances, u=u, delta1=float(np.sqrt(max(u, 0.0)) - 1.0), verified=verified,
    )


def design_via_layers(
    n: int,
    J0: Sequence[Interval],
    J1: Sequence[Interval],
    mu0: float,
    sigma: tuple | None = None,
    gamma_bound: float = 0.9995,
    n_grid_B: int = 400,
    n_grid_C: int = 400,
    n_restarts: int = 8,
    seed: int = 0,
    warm_start_alphas: np.ndarray | None = None,
) -> LayerDesignResult:
    """Directly optimize over realizable layer sequences (see module docstring).

    If sigma is None, tries all 2^len(J0) stop-band sign patterns and keeps
    the best verified result. warm_start_alphas, if given, is a length-n+1
    alpha sequence (e.g. from a known related structure, zero-padded up to
    length n+1) used as the starting point instead of small random gammas
    -- the pass/stop-band constraints are highly nonconvex in gamma and a
    generic random start is often infeasible everywhere (see commit
    history), so a physically-motivated warm start matters a lot here.
    """
    rng = np.random.default_rng(seed)
    gamma_free_init = None
    if warm_start_alphas is not None:
        gamma_free_init = np.tanh(np.asarray(warm_start_alphas, dtype=float)[:n])
    sigmas = [sigma] if sigma is not None else list(itertools.product((1, -1), repeat=len(J0)))
    best_verified, best_any = None, None
    for sig in sigmas:
        res = _solve_for_sigma(n, J0, J1, mu0, sig, gamma_bound, n_grid_B, n_grid_C, n_restarts, rng,
                                gamma_free_init=gamma_free_init)
        if res.status == "optimal" and res.verified:
            if best_verified is None or res.delta1 < best_verified.delta1:
                best_verified = res
        if best_any is None or (res.status == "optimal" and (best_any.status != "optimal" or res.delta1 < best_any.delta1)):
            best_any = res
    best = best_verified if best_verified is not None else best_any
    return best
