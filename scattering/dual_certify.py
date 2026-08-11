"""Stage 3, spec Sec. 4 / Sec. 5 rule 2 ("report dual quantities").

Manuscript Remark~\\ref{rem:certificate}: for the magnitude SDP (eq. mag),
(A) together with (D') forces `Q(0)-1=0` at a point where (A) also forces
`Q(0)-1` to attain its minimum (0) -- a double root -- so the feasible
set has EMPTY INTERIOR: Slater's condition fails, strong duality is not
automatic, and an interior-point solver's own reported primal objective
is neither `delta_mag(n)` nor a bound on it. Weak duality still suffices:
ANY dual-feasible point gives a rigorous lower bound on `delta_mag(n)`
(hence, by the magnitude relaxation's own bounding property, on every
`delta_n*(L, sigma)`), and a dual point witnessing infeasibility (a
"dual improving ray" / Farkas certificate) rigorously certifies that no
n-layer block meets the specification, regardless of period or sign
pattern. Rule 2's mandate is to emit these dual quantities explicitly,
and never report the solver's raw primal objective as if it were the
bound.

Sign convention and adjoint construction (spec Sec. 8, "where the code
has authority" -- both empirically pinned down against toy SDPs sharing
this exact block structure; not asserted from CVXPY's documentation,
since CVXPY does not commit to a single dual-sign convention across
constraint types):

For an equality constraint written as `lhs == rhs` (CVXPY canonicalizes
to `lhs - rhs == 0`), define
    y := -constraint.dual_value
    b := -(the constant part of `lhs - rhs`, i.e. the part that does not
           multiply any optimization variable)
Weak duality then gives, summed additively over every equality
constraint in the problem, `sum_i b_i . y_i <= <the true optimal value>`
-- confirmed to solver tolerance against toy Markov-Lukacs/SOS problems
with a closed-form optimum, both parities, several degrees (odd/even,
m=0..2).

For each Markov-Lukacs/Gram block (A, and each component of B, C'), the
SAME per-block y is also an explicit witness for that block's own dual
(PSD) feasibility, via the standard Lasserre moment-SOS adjoint: writing
`sigma(Y)_k = sum_{i+j=k} Y[i,j]` (poly_sdp._sos_coeffs) as a linear map,
its adjoint sends a dual vector y to the Hankel/moment matrix
`M(y)[i,j] = y[i+j]`, and dual feasibility of the block requires
`M(y) >> 0` (even-degree sigma0 block, or either linear-factor block for
odd degree) and, for the even-degree weighted term, a second Hankel
matrix built from y convolved with the interval weight polynomial
`(x-a)(b-x)` (poly_sdp._poly_mul_fixed's own adjoint is themselves a
convolution-by-the-fixed-filter, so its adjoint is correlation by the
same filter). This IS the explicit dual feasible point rule 2 asks for --
constructed independently of, and not read off from, CVXPY's own
`Y.dual_value` for the primal `Y >> 0` constraint (which lives in a
different, DOF-collapsed convention for symmetric-matrix duals -- see
scratchpad audit -- and is not used here at all).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import cvxpy as cp

from .poly_sdp import cheb_to_mono_matrix, interval_nonneg_constraints
from .sdp_design import Interval, _cheb_from_cosine_series, theta_interval_to_x


def _hankel(y: np.ndarray, size: int) -> np.ndarray:
    """M[i,j] = y[i+j], the adjoint of poly_sdp._sos_coeffs (a Hankel /
    truncated-moment matrix). Requires len(y) == 2*size - 1."""
    y = np.asarray(y, dtype=float)
    assert len(y) == 2 * size - 1, f"expected len(y)={2 * size - 1}, got {len(y)}"
    return np.array([[y[i + j] for j in range(size)] for i in range(size)])


def _localize(y: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """z[j] = sum_i weight[i]*y[i+j], the adjoint of poly_sdp._poly_mul_fixed
    (convolution by `weight`; its adjoint is correlation by the same
    filter)."""
    y = np.asarray(y, dtype=float)
    weight = np.asarray(weight, dtype=float)
    out_len = len(y) - len(weight) + 1
    return np.array([
        sum(weight[i] * y[i + j] for i in range(len(weight)))
        for j in range(out_len)
    ])


def _block_dual_matrices(y: np.ndarray, deg: int, a: float, b: float) -> list[np.ndarray]:
    """Mirrors poly_sdp.interval_nonneg_constraints' own even/odd
    branching exactly -- see module docstring for the adjoint
    derivation. Returns the matrices that must all be PSD for `y` to be
    an explicit dual-feasible witness for this block."""
    if deg % 2 == 0:
        m = deg // 2
        mats = [_hankel(y, m + 1)]
        if m >= 1:
            weight = np.array([-a * b, a + b, -1.0])  # (x-a)(b-x)
            mats.append(_hankel(_localize(y, weight), m))
        return mats
    else:
        m = (deg - 1) // 2
        z0 = _localize(y, np.array([-a, 1.0]))   # (x-a)
        z1 = _localize(y, np.array([b, -1.0]))   # (b-x)
        return [_hankel(z0, m + 1), _hankel(z1, m + 1)]


@dataclass
class BlockDualCheck:
    name: str
    min_eig: float
    feasible: bool


@dataclass
class MagnitudeSDPDualResult:
    n: int
    mu0: float
    status: str  # "dual_certified" / "dual_ray_found" / "dual_infeasible_to_certify" / "solver_failure"
    underline_delta: float | None = None      # rigorous lower bound on delta_mag(n); rule 2's own emission
    dual_ray_found: bool = False               # infeasibility certificate at degree n
    block_checks: list[BlockDualCheck] = field(default_factory=list)
    raw_primal_status: str | None = None
    raw_primal_delta: float | None = None      # DIAGNOSTIC ONLY -- never the reported bound (rule 2)
    J0: Sequence[Interval] = field(default_factory=list)
    J1: Sequence[Interval] = field(default_factory=list)


def solve_magnitude_sdp_with_duals(n: int, J0: Sequence[Interval], J1: Sequence[Interval], mu0: float,
                                    solver: str = "CLARABEL", tol: float = 1e-6,
                                    **solver_kwargs) -> MagnitudeSDPDualResult:
    """Solves the same magnitude SDP as sdp_design.design_sdp_magnitude
    (manuscript eq. mag), but additionally extracts, for every equality
    constraint, an explicit dual-feasible point and reports
    `underline_delta` from it (optimal case) or checks it as a dual
    improving ray (infeasible case) -- see module docstring. CLARABEL
    exposes `.dual_value` on every constraint in both the optimal and
    infeasible cases (confirmed empirically), so the same extraction
    code handles both branches.
    """
    T = cheb_to_mono_matrix(n)
    f = cp.Variable(n + 1)
    delta = cp.Variable()
    e0 = np.zeros(n + 1)
    e0[0] = 1.0
    scale = np.concatenate([[1.0], 2.0 * np.ones(n)])

    delta_nonneg = (delta >= 0.0)
    eq_D = (cp.sum(cp.multiply(scale, f)) == 1.0)  # (D'): Q(0) = 1
    constraints = [delta_nonneg, eq_D]

    blocks = []  # (name, deg, a, b, equality_constraint, b_vec)

    consA, _ = interval_nonneg_constraints(T @ _cheb_from_cosine_series(f - e0, n), n, -1.0, 1.0)
    constraints += consA
    b_A = np.zeros(n + 1)
    b_A[0] = 1.0
    blocks.append(("A", n, -1.0, 1.0, consA[-1], b_A))

    for i, (gamma, delta_hi) in enumerate(J1):
        xlo, xhi = theta_interval_to_x(gamma, delta_hi)
        g_B = (delta + 1.0) * e0 - f
        consB, _ = interval_nonneg_constraints(T @ _cheb_from_cosine_series(g_B, n), n, xlo, xhi)
        constraints += consB
        b_B = np.zeros(n + 1)
        b_B[0] = -1.0
        blocks.append((f"B{i}", n, xlo, xhi, consB[-1], b_B))

    coshmu0_sq = float(np.cosh(mu0) ** 2)
    for i, (u, v) in enumerate(J0):
        xlo, xhi = theta_interval_to_x(u, v)
        g_C = f - coshmu0_sq * e0
        consC, _ = interval_nonneg_constraints(T @ _cheb_from_cosine_series(g_C, n), n, xlo, xhi)
        constraints += consC
        b_C = np.zeros(n + 1)
        b_C[0] = coshmu0_sq
        blocks.append((f"C{i}", n, xlo, xhi, consC[-1], b_C))

    problem = cp.Problem(cp.Minimize(delta), constraints)
    try:
        problem.solve(solver=solver, **solver_kwargs)
    except cp.error.SolverError:
        return MagnitudeSDPDualResult(n=n, mu0=mu0, status="solver_failure", J0=list(J0), J1=list(J1))

    raw_status = problem.status
    if raw_status not in ("optimal", "optimal_inaccurate", "infeasible", "infeasible_inaccurate"):
        return MagnitudeSDPDualResult(n=n, mu0=mu0, status="solver_failure", raw_primal_status=raw_status,
                                       J0=list(J0), J1=list(J1))

    if eq_D.dual_value is None or any(blk[4].dual_value is None for blk in blocks):
        return MagnitudeSDPDualResult(n=n, mu0=mu0, status="solver_failure", raw_primal_status=raw_status,
                                       J0=list(J0), J1=list(J1))

    y_D = -float(eq_D.dual_value)
    underline_delta = 1.0 * y_D  # b_D = 1.0, (D')'s own RHS
    block_checks = []
    all_feasible = True
    for name, deg, a, b, eq, b_vec in blocks:
        y = -np.asarray(eq.dual_value, dtype=float).flatten()
        underline_delta += float(b_vec @ y)
        mats = _block_dual_matrices(y, deg, a, b)
        min_eig = min(float(np.linalg.eigvalsh(M).min()) for M in mats)
        feasible = min_eig >= -tol
        all_feasible = all_feasible and feasible
        block_checks.append(BlockDualCheck(name=name, min_eig=min_eig, feasible=feasible))

    raw_delta = (float(delta.value)
                 if raw_status in ("optimal", "optimal_inaccurate") and delta.value is not None
                 else None)

    if raw_status in ("infeasible", "infeasible_inaccurate"):
        dual_ray_found = all_feasible and underline_delta > tol
        status = "dual_ray_found" if dual_ray_found else "dual_infeasible_to_certify"
        return MagnitudeSDPDualResult(n=n, mu0=mu0, status=status, dual_ray_found=dual_ray_found,
                                       block_checks=block_checks, raw_primal_status=raw_status,
                                       J0=list(J0), J1=list(J1))

    if not all_feasible:
        # Rule 2: never report the primal objective as the bound -- if
        # the explicit dual witness fails its own feasibility check, we
        # do NOT fall back to raw_delta as underline_delta.
        return MagnitudeSDPDualResult(n=n, mu0=mu0, status="dual_infeasible_to_certify",
                                       block_checks=block_checks, raw_primal_status=raw_status,
                                       raw_primal_delta=raw_delta, J0=list(J0), J1=list(J1))

    return MagnitudeSDPDualResult(n=n, mu0=mu0, status="dual_certified", underline_delta=underline_delta,
                                   block_checks=block_checks, raw_primal_status=raw_status,
                                   raw_primal_delta=raw_delta, J0=list(J0), J1=list(J1))
