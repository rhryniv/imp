"""Markov-Lukacs SOS representability of "p(x) >= 0 on [a,b]", encoded
directly in the CHEBYSHEV basis (not monomial -- badly conditioned by
n~6), via T_i*T_j = 0.5*(T_{i+j}+T_{|i-j|}).

Convention: a "cheb coeff vector" c of length d+1 represents
p(x) = sum_{k=0}^d c_k T_k(x).
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp


def _stack_buckets(buckets: list[list]) -> cp.Expression:
    """Each bucket accumulates the (possibly zero, possibly empty)
    contributions to one output coefficient; an empty bucket means that
    coefficient is identically zero (no (i,j)/(m,k) pair reaches it),
    not a bug -- e.g. weight_interval_product(-1,1) has a genuine zero
    middle coefficient (a+b=0), which then leaves some product indices
    with no contributing pair at all."""
    return cp.hstack([cp.sum(cp.hstack(b)) if len(b) > 1 else (b[0] if b else cp.Constant(0.0))
                       for b in buckets])


def sos_coeffs_cheb(G: cp.Expression, r: int) -> cp.Expression:
    """Chebyshev coefficients (length 2r+1) of sigma(x) = z(x)^T G z(x),
    z=(T_0,...,T_r)^T, via T_i*T_j = 0.5*(T_{i+j}+T_{|i-j|}). Built by
    accumulating each (i,j) pair's contribution directly (cvxpy affine
    sum), not via a fixed matrix -- G is a cvxpy Variable."""
    buckets: list[list] = [[] for _ in range(2 * r + 1)]
    for i in range(r + 1):
        for j in range(r + 1):
            buckets[i + j].append(0.5 * G[i, j])
            buckets[abs(i - j)].append(0.5 * G[i, j])
    return _stack_buckets(buckets)


def cheb_poly_mul_fixed(weight: np.ndarray, var_expr: cp.Expression, q_deg: int) -> cp.Expression:
    """Chebyshev coefficients of weight(x)*var(x), weight a KNOWN numeric
    Chebyshev-coeff vector (len p_deg+1), var_expr a cvxpy Chebyshev-coeff
    expression (len q_deg+1), via T_i*T_j=0.5*(T_{i+j}+T_{|i-j|})."""
    p_deg = len(weight) - 1
    out_len = p_deg + q_deg + 1
    buckets: list[list] = [[] for _ in range(out_len)]
    for m, wm in enumerate(weight):
        if wm == 0.0:
            continue
        for k in range(q_deg + 1):
            buckets[m + k].append(0.5 * wm * var_expr[k])
            buckets[abs(m - k)].append(0.5 * wm * var_expr[k])
    return _stack_buckets(buckets)


def weight_x_minus_a(a: float) -> np.ndarray:
    """Chebyshev coeffs of (x-a) = T_1 - a*T_0."""
    return np.array([-a, 1.0])


def weight_b_minus_x(b: float) -> np.ndarray:
    """Chebyshev coeffs of (b-x) = b*T_0 - T_1."""
    return np.array([b, -1.0])


def weight_interval_product(a: float, b: float) -> np.ndarray:
    """Chebyshev coeffs of (x-a)(b-x) = -x^2+(a+b)x-ab
    = (-ab-0.5)*T_0 + (a+b)*T_1 - 0.5*T_2   (x^2=(T_0+T_2)/2)."""
    return np.array([-a * b - 0.5, a + b, -0.5])


def markov_lukacs_cheb(coeff_expr: cp.Expression, deg: int, a: float, b: float):
    """Constraints forcing p(x)=sum coeff_expr[k] T_k(x) >= 0 on [a,b],
    Chebyshev-basis Markov-Lukacs. Returns (constraints, gram_vars,
    raw_expr) where raw_expr = coeff_expr - rhs(Gram vars) is the SAME
    affine expression wrapped as `raw_expr == 0` inside `constraints` --
    kept unwrapped too so callers can extract its constant part (for the
    explicit dual bound) without parsing cvxpy internals."""
    assert b > a
    constraints = []
    gram_vars = []
    if deg % 2 == 0:
        r = deg // 2
        Y0 = cp.Variable((r + 1, r + 1), symmetric=True)
        constraints.append(Y0 >> 0)
        gram_vars.append(Y0)
        sigma0 = sos_coeffs_cheb(Y0, r)
        if r >= 1:
            Y1 = cp.Variable((r, r), symmetric=True)
            constraints.append(Y1 >> 0)
            gram_vars.append(Y1)
            sigma1 = sos_coeffs_cheb(Y1, r - 1)
            weight = weight_interval_product(a, b)
            rhs = sigma0 + cheb_poly_mul_fixed(weight, sigma1, deg - 2)
        else:
            rhs = sigma0
    else:
        r = (deg - 1) // 2
        Y0 = cp.Variable((r + 1, r + 1), symmetric=True)
        Y1 = cp.Variable((r + 1, r + 1), symmetric=True)
        constraints += [Y0 >> 0, Y1 >> 0]
        gram_vars += [Y0, Y1]
        sigma0 = sos_coeffs_cheb(Y0, r)
        sigma1 = sos_coeffs_cheb(Y1, r)
        t0 = cheb_poly_mul_fixed(weight_x_minus_a(a), sigma0, deg - 1)
        t1 = cheb_poly_mul_fixed(weight_b_minus_x(b), sigma1, deg - 1)
        rhs = t0 + t1
    raw_expr = coeff_expr - rhs
    constraints.append(raw_expr == 0)
    return constraints, gram_vars, raw_expr


def hankel_toeplitz_dual(y: np.ndarray, size: int) -> np.ndarray:
    """Adjoint of sos_coeffs_cheb: M(y)[i,j] = 0.5*(y[i+j]+y[|i-j|]),
    the Chebyshev-basis dual moment matrix (Hankel+Toeplitz)/2."""
    y = np.asarray(y, dtype=float)
    M = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            M[i, j] = 0.5 * (y[i + j] + y[abs(i - j)])
    return M


def cheb_localize_dual(y: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Adjoint of cheb_poly_mul_fixed (for FIXED weight): z_k =
    sum_m weight_m * 0.5*(y[m+k]+y[|m-k|])."""
    y = np.asarray(y, dtype=float)
    weight = np.asarray(weight, dtype=float)
    out_len = len(y) - (len(weight) - 1)
    z = np.zeros(out_len)
    for k in range(out_len):
        s = 0.0
        for m, wm in enumerate(weight):
            s += wm * 0.5 * (y[m + k] + y[abs(m - k)])
        z[k] = s
    return z


def dual_matrices_cheb(y: np.ndarray, deg: int, a: float, b: float) -> list[np.ndarray]:
    """Explicit dual-feasibility witness matrices for markov_lukacs_cheb's
    own even/odd branches -- must all be PSD for y to be dual feasible."""
    if deg % 2 == 0:
        r = deg // 2
        mats = [hankel_toeplitz_dual(y, r + 1)]
        if r >= 1:
            weight = weight_interval_product(a, b)
            z = cheb_localize_dual(y, weight)
            mats.append(hankel_toeplitz_dual(z, r))
        return mats
    else:
        r = (deg - 1) // 2
        z0 = cheb_localize_dual(y, weight_x_minus_a(a))
        z1 = cheb_localize_dual(y, weight_b_minus_x(b))
        return [hankel_toeplitz_dual(z0, r + 1), hankel_toeplitz_dual(z1, r + 1)]
