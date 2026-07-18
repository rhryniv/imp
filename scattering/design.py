"""Autocorrelation-domain SDP for the single-block pass-band design problem.

This is the "simpler" of the two formulations discussed for the design
problem (DP) / program (5.16)-(5.19) of the paper: it enforces constraints
(A) admissibility and (B) pass-band flatness, plus the normalisation (D),
but *not* the stop-band constraint (C) -- see Remark 5.10 ("autocorrelation
alternative"), which is exactly the approach used for the paper's own
worked example (Section 6.4).

Working variables are the autocorrelation coefficients
    c_l = sum_{m=0}^{n-l} a_m a_{m+l},   l = 0, ..., n,
equivalently the Chebyshev coefficients of G(x) = |q~_1(theta)|^2 (x=cos
theta): G(x) = sum_l c_l T_l(x).  Both (A) and (B) are affine in c, and
G(1) = sum_l c_l, so the normalisation q~_1(0)=1 is exactly the linear
constraint sum(c) = 1.

Constraint (B), "|q~_1|^2 <= (1+delta1)^2 on J1", is *not* jointly convex
in (c, delta1) if you literally keep the square -- (1+delta1)^2 is convex
in delta1 while the inequality needs it as an upper bound, which flips
convexity.  The fix is the standard epigraph trick: introduce u = (1+delta1)^2
directly as a free real variable, require G(x) <= u on J1 (now affine),
minimize u, and report delta1 = sqrt(u*) - 1 afterwards.  This gives the
*exact* same optimum as minimising delta1 (u -> (1+delta1)^2 is a strictly
increasing reparametrisation for delta1 >= 0), with no bisection required.

After solving, call spectral_factor.fejer_riesz_min_phase(c) to recover a
single-block coefficient vector a_m, and design.check_stopband(...) to
verify constraint (C) post hoc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import cvxpy as cp

from .poly_sdp import cheb_to_mono_matrix, interval_nonneg_constraints

Interval = tuple[float, float]


def theta_interval_to_x(gamma: float, delta: float) -> Interval:
    """theta-interval [gamma,delta] subset [0,pi] -> x-interval [cos delta, cos gamma]."""
    if not (0.0 <= gamma < delta <= np.pi + 1e-12):
        raise ValueError(f"expected 0 <= gamma < delta <= pi, got ({gamma}, {delta})")
    return float(np.cos(delta)), float(np.cos(gamma))


@dataclass
class DesignResult:
    n: int
    status: str
    c: np.ndarray | None = None          # autocorrelation / Chebyshev coeffs of G, length n+1
    u: float | None = None               # = (1+delta1)^2
    delta1: float | None = None          # pass-band ripple in |q~_1|
    problem: cp.Problem | None = None
    J0: Sequence[Interval] = field(default_factory=list)
    J1: Sequence[Interval] = field(default_factory=list)


def design_filter(
    n: int,
    J0: Sequence[Interval],
    J1: Sequence[Interval],
    solver: str = "CLARABEL",
    **solver_kwargs,
) -> DesignResult:
    """Solve the autocorrelation-domain relaxation of the design problem.

    J0, J1 : lists of (alpha, beta) / (gamma, delta) pairs in the
        theta = 2*k*h variable, each a subinterval of [0, pi].  J0 is not
        used by this SDP (constraint (C) is checked post hoc, see
        `check_stopband`) but is carried along in the result for
        convenience / bookkeeping.

    Returns a DesignResult. status is cvxpy's problem.status; on success
    it is "optimal".
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    T = cheb_to_mono_matrix(n)
    c = cp.Variable(n + 1)
    u = cp.Variable()
    e0 = np.zeros(n + 1)
    e0[0] = 1.0

    constraints = [u >= 1.0, cp.sum(c) == 1.0]

    # (A) G(x) - 1 >= 0 on x in [-1, 1]  (equivalent to |q~_1(theta)|^2 >= 1 for all real theta)
    consA, _ = interval_nonneg_constraints(T @ (c - e0), n, -1.0, 1.0)
    constraints += consA

    # (B) u - G(x) >= 0 on each pass-band component (x-image of J1)
    for gamma, delta in J1:
        xlo, xhi = theta_interval_to_x(gamma, delta)
        consB, _ = interval_nonneg_constraints(T @ (u * e0 - c), n, xlo, xhi)
        constraints += consB

    problem = cp.Problem(cp.Minimize(u), constraints)
    problem.solve(solver=solver, **solver_kwargs)

    result = DesignResult(n=n, status=problem.status, problem=problem, J0=list(J0), J1=list(J1))
    if problem.status in ("optimal", "optimal_inaccurate"):
        result.c = c.value
        result.u = float(u.value)
        result.delta1 = float(np.sqrt(max(result.u, 0.0)) - 1.0)
    return result


def degree_scan(n_values: Sequence[int], J0: Sequence[Interval], J1: Sequence[Interval], **kwargs):
    """Solve design_filter for a range of n, return dict n -> DesignResult."""
    return {n: design_filter(n, J0, J1, **kwargs) for n in n_values}
