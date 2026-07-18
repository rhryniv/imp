"""The full, jointly-optimal SDP of Appendix C (eq. 5.22-5.24): constraints
(A) admissibility, (B) pass-band flatness and (C) stop-band depth are all
enforced *together*, over the same coefficient vector a, via the rank
relaxation

    A ~ a a^T      relaxed to      [[A, a], [a^T, 1]] >> 0.

Constraint (C) is linear in a directly (sigma_j * sum_m a_m cos(m*theta) >=
cosh(mu0) on each stop component), constraints (A)/(B) are linear in the
"lifted autocorrelation" c_l(A) = sum of the l-th diagonal of A -- which
equals the true autocorrelation sum_m a_m a_{m+l} exactly when A = a a^T,
i.e. when the relaxation is tight (rank(A) = 1).  We check tightness
numerically after solving (eigenvalue ratio of A, and a direct pointwise
verification of (A)/(B)/(C) using the realized vector a, not the relaxed A)
-- see Remark 5.15 in the paper.

The sign pattern sigma_j in (C) ("which band edge the discriminant enters
the j-th stop component through") is a discrete choice, not a continuous
SDP variable. With m0 stop components there are 2^m0 patterns; for the
small m0 this design problem is meant for, we just try all of them and
keep the best feasible one.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import cvxpy as cp

from .poly_sdp import cheb_to_mono_matrix, interval_nonneg_constraints
from .design import Interval, theta_interval_to_x
from .transmission import kappa_B
from .spectral_factor import evaluate_G_from_a


def _autocorr_from_A(A: cp.Expression, n: int) -> cp.Expression:
    """c_l(A) = sum of the l-th diagonal of A, l=0..n (equals sum_m a_m a_{m+l} when A=a a^T)."""
    terms = [cp.trace(A)]
    for l in range(1, n + 1):
        diag_terms = [A[l + i, i] for i in range(n + 1 - l)]
        terms.append(cp.sum(cp.hstack(diag_terms)) if len(diag_terms) > 1 else diag_terms[0])
    return cp.hstack(terms)


@dataclass
class FullDesignResult:
    n: int
    mu0: float
    status: str
    sigma: tuple | None = None
    a: np.ndarray | None = None
    A: np.ndarray | None = None
    u: float | None = None
    delta1: float | None = None
    tightness_ratio: float | None = None   # lambda_2 / lambda_1 of A (0 = perfectly rank-1)
    verified: bool | None = None           # pointwise check of (A),(B),(C) using the realized a
    problem: cp.Problem | None = None
    J0: Sequence[Interval] = field(default_factory=list)
    J1: Sequence[Interval] = field(default_factory=list)


def _solve_for_sigma(n, J0, J1, mu0, sigma, T, solver, solver_kwargs):
    a = cp.Variable(n + 1)
    A = cp.Variable((n + 1, n + 1), symmetric=True)
    u = cp.Variable()
    e0 = np.zeros(n + 1)
    e0[0] = 1.0

    lift = cp.bmat([[A, cp.reshape(a, (n + 1, 1), order="C")],
                    [cp.reshape(a, (1, n + 1), order="C"), np.array([[1.0]])]])
    constraints = [lift >> 0, u >= 1.0, cp.sum(a) == 1.0]

    c = _autocorr_from_A(A, n)

    consA, _ = interval_nonneg_constraints(T @ (c - e0), n, -1.0, 1.0)
    constraints += consA

    for gamma, delta in J1:
        xlo, xhi = theta_interval_to_x(gamma, delta)
        consB, _ = interval_nonneg_constraints(T @ (u * e0 - c), n, xlo, xhi)
        constraints += consB

    coshmu0 = float(np.cosh(mu0))
    for (alpha, beta), sig in zip(J0, sigma):
        xlo, xhi = theta_interval_to_x(alpha, beta)
        stop_cheb = sig * a - coshmu0 * e0
        consC, _ = interval_nonneg_constraints(T @ stop_cheb, n, xlo, xhi)
        constraints += consC

    problem = cp.Problem(cp.Minimize(u), constraints)
    problem.solve(solver=solver, **solver_kwargs)

    result = FullDesignResult(n=n, mu0=mu0, status=problem.status, sigma=sigma,
                               problem=problem, J0=list(J0), J1=list(J1))
    if problem.status in ("optimal", "optimal_inaccurate"):
        result.a = a.value
        result.A = A.value
        result.u = float(u.value)
        result.delta1 = float(np.sqrt(max(result.u, 0.0)) - 1.0)
        eigvals = np.sort(np.linalg.eigvalsh(result.A))[::-1]
        result.tightness_ratio = float(eigvals[1] / eigvals[0]) if eigvals[0] > 1e-12 else None
        result.verified = verify_design(result.a, J0, J1, mu0, sigma, result.u)
    return result


def design_filter_full(
    n: int,
    J0: Sequence[Interval],
    J1: Sequence[Interval],
    mu0: float,
    sigma: tuple | None = None,
    solver: str = "CLARABEL",
    **solver_kwargs,
) -> FullDesignResult:
    """Solve the full lifted SDP (A)+(B)+(C)+(D). If sigma is None, tries all
    2^len(J0) sign patterns and returns the feasible one with smallest u
    (largest pass-band flatness); returns the best *attempted* result (by
    status) if none are feasible."""
    T = cheb_to_mono_matrix(n)
    if sigma is not None:
        return _solve_for_sigma(n, J0, J1, mu0, sigma, T, solver, solver_kwargs)

    best = None
    for sig in itertools.product((1, -1), repeat=len(J0)):
        res = _solve_for_sigma(n, J0, J1, mu0, sig, T, solver, solver_kwargs)
        if res.status in ("optimal", "optimal_inaccurate"):
            if best is None or res.u < best.u:
                best = res
        elif best is None:
            best = res  # keep *something* to report if nothing is feasible
    return best


def verify_design(a: np.ndarray, J0, J1, mu0: float, sigma, u: float, n_grid: int = 4000, tol: float = 1e-6) -> bool:
    """Direct pointwise check of (A),(B),(C) using the realized vector a
    (not the relaxed lift), the ultimate tightness/correctness certificate."""
    theta_full = np.linspace(0.0, np.pi, 20 * n_grid)
    G = evaluate_G_from_a(a, theta_full)
    if np.min(G) < 1.0 - tol:
        return False
    for gamma, delta in J1:
        th = np.linspace(gamma, delta, n_grid)
        if np.max(evaluate_G_from_a(a, th)) > u + tol:
            return False
    for (alpha, beta), sig in zip(J0, sigma):
        th = np.linspace(alpha, beta, n_grid)
        kap = kappa_B(a, th)
        if np.min(sig * kap) < np.cosh(mu0) - tol:
            return False
    return True
