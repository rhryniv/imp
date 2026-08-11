"""Semidefinite-representable nonnegativity constraints for real polynomials.

Every trigonometric object in the paper (the discriminant, |q~_1|^2, the
stop-band inequality) is, after the substitution x = cos(theta), a plain
real polynomial in x of degree <= n.  So instead of working with the
trigonometric Gram/Markov-Lukacs machinery of Appendix B directly (which
mixes cosine harmonics and is easy to get an off-by-one wrong in), we do
everything in the algebraic (monomial) domain, where the classical
Markov-Lukacs / sum-of-squares representation of "polynomial nonnegative
on an interval" is completely standard and unambiguous.  Chebyshev
coefficients are converted to monomial coefficients via a fixed linear
change of basis (numpy.polynomial.chebyshev), so cvxpy only ever sees
plain linear/affine expressions.

Convention: a coefficient vector ``coeffs`` of length ``d + 1`` represents
    p(x) = sum_{k=0}^{d} coeffs[k] * x**k        (increasing powers).
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp
from numpy.polynomial import chebyshev as C


def cheb_to_mono_matrix(n: int) -> np.ndarray:
    """Matrix T (n+1)x(n+1) with mono_coeffs = T @ cheb_coeffs.

    cheb_coeffs are the coefficients of p(x) = sum_l c_l * T_l(x)
    (T_l = Chebyshev polynomial of the first kind), mono_coeffs the
    coefficients of the same polynomial in the power basis.
    """
    T = np.zeros((n + 1, n + 1))
    for l in range(n + 1):
        e = np.zeros(n + 1)
        e[l] = 1.0
        mono = C.cheb2poly(e)
        T[: len(mono), l] = mono
    return T


def _sos_coeffs(Y: cp.Expression, m: int) -> cp.Expression:
    """Coefficient vector (length 2m+1) of v(x)^T Y v(x), v=(1,x,...,x^m)."""
    terms = []
    for k in range(2 * m + 1):
        i_lo, i_hi = max(0, k - m), min(m, k)
        row = [Y[i, k - i] for i in range(i_lo, i_hi + 1)]
        terms.append(cp.sum(cp.hstack(row)) if len(row) > 1 else row[0])
    return cp.hstack(terms)


def _poly_mul_fixed(fixed: np.ndarray, var_expr: cp.Expression, q_deg: int) -> cp.Expression:
    """Coefficients of fixed(x) * var(x), fixed known (len p+1), var length q_deg+1."""
    p_deg = len(fixed) - 1
    out_len = p_deg + q_deg + 1
    M = np.zeros((out_len, q_deg + 1))
    for i, fi in enumerate(fixed):
        for j in range(q_deg + 1):
            M[i + j, j] += fi
    return M @ var_expr


def interval_nonneg_constraints(coeff_expr: cp.Expression, deg: int, a: float, b: float):
    """Constraints forcing p(x) = sum coeff_expr[k] x^k  >= 0  for x in [a, b].

    Classical Markov-Lukacs / SOS representation (deg = highest possible
    power present, coeff_expr must have length deg+1):
      deg = 2m   : p = sigma0 + (x-a)(b-x) * sigma1,
                   sigma0 SOS of degree <= 2m   (Gram size m+1),
                   sigma1 SOS of degree <= 2m-2 (Gram size m).
      deg = 2m+1 : p = (x-a) * sigma0 + (b-x) * sigma1,
                   sigma0, sigma1 SOS of degree <= 2m (Gram size m+1 each).
    Returns (constraints, gram_vars) so callers can inspect the Gram
    matrices afterwards (e.g. for a rank/tightness check) if desired.

    Spec Sec. 5 rule 1 ("odd-degree padding"): the odd-degree branch above
    is the direct/standard Markov-Lukacs form for odd degree (each SOS
    term carries its own degree-1 factor, (x-a) or (b-x), rather than a
    single degree-2m SOS padded up to 2m+1) -- mathematically equivalent
    to padding, not a truncated/even-only implementation. Empirically
    audited (tests/test_stage3_poly_sdp.py) against polynomials with known
    nonnegativity status on an interval, both parities, including
    boundary-touching and interior-double-root edge cases: no case found
    where this returns a feasible/infeasible verdict inconsistent with the
    true pointwise sign. The degree-parity assertion below guards the
    other half of rule 1 -- a coeff_expr/deg length mismatch at the call
    site, which is the actual "returns a number that is not a bound,
    without erroring" failure mode the rule describes.
    """
    assert b > a
    assert coeff_expr.shape == (deg + 1,), (
        f"coeff_expr has shape {coeff_expr.shape}, expected ({deg + 1},) for deg={deg} "
        "-- spec Sec. 5 rule 1: a length/degree mismatch here is exactly the silent, "
        "non-erroring failure mode rule 1 warns about, so this is asserted rather than "
        "left to fail downstream (or not fail at all)."
    )
    constraints = []
    gram_vars = []
    if deg % 2 == 0:
        m = deg // 2
        Y0 = cp.Variable((m + 1, m + 1), symmetric=True)
        constraints.append(Y0 >> 0)
        gram_vars.append(Y0)
        sigma0 = _sos_coeffs(Y0, m)
        if m >= 1:
            Y1 = cp.Variable((m, m), symmetric=True)
            constraints.append(Y1 >> 0)
            gram_vars.append(Y1)
            sigma1 = _sos_coeffs(Y1, m - 1)
            weight = np.array([-a * b, a + b, -1.0])  # (x-a)(b-x)
            rhs = sigma0 + _poly_mul_fixed(weight, sigma1, deg - 2)
        else:
            rhs = sigma0
        constraints.append(coeff_expr == rhs)
    else:
        m = (deg - 1) // 2
        Y0 = cp.Variable((m + 1, m + 1), symmetric=True)
        Y1 = cp.Variable((m + 1, m + 1), symmetric=True)
        constraints += [Y0 >> 0, Y1 >> 0]
        gram_vars += [Y0, Y1]
        sigma0 = _sos_coeffs(Y0, m)
        sigma1 = _sos_coeffs(Y1, m)
        t0 = _poly_mul_fixed(np.array([-a, 1.0]), sigma0, deg - 1)   # (x-a)*sigma0
        t1 = _poly_mul_fixed(np.array([b, -1.0]), sigma1, deg - 1)   # (b-x)*sigma1
        constraints.append(coeff_expr == t0 + t1)
    return constraints, gram_vars
