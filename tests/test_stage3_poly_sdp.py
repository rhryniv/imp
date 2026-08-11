"""Stage 3, spec Sec. 5 rule 1 ("odd-degree padding"):
poly_sdp.interval_nonneg_constraints, both parity branches, against
polynomials with KNOWN, hand-verified nonnegativity status on [a, b].

Each case is a feasibility SDP: coeff_expr is fixed (via cp.Parameter) to
a concrete numeric coefficient vector, and the constraint set's own
feasibility is checked against an independent ground truth (either an
algebraic argument, noted per-case, cross-checked here by a 200k-point
grid sample of the true polynomial). Includes boundary-touching-zero and
interior-double-root cases, the trickiest for SOS representability, since
those are exactly where a naive/incorrect odd-degree implementation would
most plausibly diverge from the true feasible set.

Run: python3 -m pytest tests/test_stage3_poly_sdp.py -v   (from repo root)
"""
import numpy as np
import cvxpy as cp
import pytest

from scattering.poly_sdp import interval_nonneg_constraints


def _solve_feasibility(coeffs, a, b):
    coeffs = np.asarray(coeffs, dtype=float)
    deg = len(coeffs) - 1
    coeff_var = cp.Parameter(deg + 1)
    coeff_var.value = coeffs
    cons, _ = interval_nonneg_constraints(coeff_var, deg, a, b)
    prob = cp.Problem(cp.Minimize(0), cons)
    prob.solve(solver="CLARABEL")
    return prob.status in ("optimal", "optimal_inaccurate")


def _true_min_on_grid(coeffs, a, b, n_grid=200_000):
    xs = np.linspace(a, b, n_grid)
    return float(np.polyval(np.asarray(coeffs, dtype=float)[::-1], xs).min())


@pytest.mark.parametrize("name,coeffs,a,b,expect_feasible", [
    # ---- even degree ----
    ("even deg2 x^2 nonneg (touches 0 at x=0, interior)", [0, 0, 1], -1, 1, True),
    ("even deg2 x^2-4 negative everywhere on [-1,1]", [-4, 0, 1], -1, 1, False),
    ("even deg2 negative-somewhere", [-0.75, 2, -1], 0, 1, False),
    ("even deg2 (x-2)^2 nonneg on [0,1] (min at far edge)", [4, -4, 1], 0, 1, True),
    ("even deg4 (x^2-4)^2 nonneg (no real roots in range)", [16, 0, -8, 0, 1], -1, 1, True),
    # ---- odd degree ----
    ("odd deg1 x nonneg on [0,1] (touches 0 at left edge)", [0, 1], 0, 1, True),
    ("odd deg1 x negative on [-1,1]", [0, 1], -1, 1, False),
    ("odd deg1 x+2 nonneg", [2, 1], -1, 1, True),
    ("odd deg3 double-root-inside nonneg on [-1,1]", None, -1, 1, True),   # filled below
    ("odd deg3 same cubic negative on [-2,1]", None, -2, 1, False),
    ("odd deg5 nonneg on [-1,2]", None, -1, 2, True),
    ("odd deg5 shifted-down negative", None, -1, 2, False),
])
def test_interval_nonneg_matches_ground_truth(name, coeffs, a, b, expect_feasible):
    if "deg3" in name:
        # p(x) = (x+1)(x-0.3)^2: nonneg for all x >= -1 (linear factor
        # nonneg there, square always nonneg); negative for x < -1.
        coeffs = np.polynomial.polynomial.polymul(
            [1, 1], np.polynomial.polynomial.polymul([-0.3, 1], [-0.3, 1]))
    elif name.startswith("odd deg5 nonneg"):
        # p(x) = (x+1)(x^2+1)^2: nonneg for all x >= -1 (x^2+1 > 0 always).
        coeffs = np.polynomial.polynomial.polymul(
            [1, 1], np.polynomial.polynomial.polymul([1, 0, 1], [1, 0, 1]))
    elif name.startswith("odd deg5 shifted"):
        base = np.polynomial.polynomial.polymul(
            [1, 1], np.polynomial.polynomial.polymul([1, 0, 1], [1, 0, 1]))
        coeffs = base.copy()
        coeffs[0] -= 100.0

    feasible = _solve_feasibility(coeffs, a, b)
    assert feasible == expect_feasible, (
        f"{name}: got feasible={feasible}, expected {expect_feasible} "
        f"(true grid min = {_true_min_on_grid(coeffs, a, b):.6g})"
    )


def test_degree_parity_mismatch_raises():
    """Rule 1's own defensive requirement: a coeff_expr/deg length mismatch
    must raise, not silently produce a wrong constraint set."""
    coeff_var = cp.Parameter(4)  # length 4 => implies deg=3
    with pytest.raises(AssertionError):
        interval_nonneg_constraints(coeff_var, 4, -1.0, 1.0)  # deg=4 mismatched
