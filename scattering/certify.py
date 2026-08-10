"""Exact certification via rootfinding (manuscript Sec. 6.3, "Certification";
spec Sec. 6). Optimisation (design_direct, design_sdp_magnitude) runs on a
grid, but the numbers finally reported here do not depend on it: Q and
kappa are trigonometric polynomials, so their extrema on any interval are
attained either at a root of the derivative *inside* the interval, or at
an endpoint (the typical case for kappa across a gap, where it is
monotone -- omitting endpoints gives a wrong depth). Both are computed
exactly, roots via companion-matrix eigenvalues (numpy.roots), no
grid search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .forward import a_from_alphas, autocorr, kappa_B, q1_abs_sq

Interval = tuple[float, float]


def _deriv_poly_coeffs(coeffs: np.ndarray) -> np.ndarray:
    """Given g(theta) = sum_{m=0}^n coeffs[m] * cos(m*theta) (kappa's own
    form) or, via the autocorrelation, Q's, this returns the coefficients
    (degree 0..2n, increasing power of z) of P(z) := z^n * g'(theta) / i,
    where g'(theta) = -sum_{m=1}^n m*coeffs[m]*sin(m*theta) and
    z=exp(i*theta). Roots of g'(theta)=0 on the real line correspond
    exactly to roots of P(z) with |z|=1, theta=arg(z) (verified: P is
    real-coefficient and antisymmetric about degree n, P[n+m]=-P[n-m],
    which is exactly what a sine-polynomial's z^n-cleared form must be).
    """
    n = len(coeffs) - 1
    P = np.zeros(2 * n + 1)
    for m in range(1, n + 1):
        c = m * coeffs[m]
        P[n + m] += c
        P[n - m] -= c
    return P


def _extremal_thetas(deriv_coeffs: np.ndarray, lo: float, hi: float,
                      circle_tol: float = 1e-6) -> np.ndarray:
    """Roots of the derivative (via deriv_coeffs, see _deriv_poly_coeffs)
    that correspond to real theta strictly inside (lo, hi). Extraneous
    roots of the z^n-cleared polynomial that do not lie on the unit
    circle (an artifact of clearing negative powers, not real critical
    points) are discarded by the |z|~=1 filter."""
    if not np.any(deriv_coeffs):
        return np.array([])
    nz = np.flatnonzero(deriv_coeffs)
    trimmed = deriv_coeffs[nz[0]: nz[-1] + 1]
    if len(trimmed) <= 1:
        return np.array([])
    roots = np.roots(trimmed[::-1])
    on_circle = roots[np.abs(np.abs(roots) - 1.0) < circle_tol]
    thetas = np.angle(on_circle)
    candidates = np.concatenate([thetas, thetas - 2 * np.pi, thetas + 2 * np.pi])
    inside = candidates[(candidates > lo + 1e-12) & (candidates < hi - 1e-12)]
    return np.array(sorted(set(np.round(inside, 12))))


def _extremize(coeffs: np.ndarray, intervals: Sequence[Interval], eval_fn, mode: str) -> float:
    """max/min of eval_fn(theta) over the union of intervals, exactly:
    eval_fn evaluated only at the interval endpoints and the exact
    critical points of the trig polynomial whose coefficients (in the
    "coeffs" cos-series convention) are given."""
    deriv_coeffs = _deriv_poly_coeffs(coeffs)
    best = None
    for lo, hi in intervals:
        thetas = _extremal_thetas(deriv_coeffs, lo, hi)
        candidates = np.concatenate([thetas, [lo, hi]])
        vals = eval_fn(candidates)
        v = float(np.max(vals) if mode == "max" else np.min(vals))
        best = v if best is None else (max(best, v) if mode == "max" else min(best, v))
    return best


@dataclass
class CertifyResult:
    delta: float                       # exact max_{I1}(Q-1) (manuscript eq. 6.6's own delta)
    mu_min: float                      # exact arccosh(d(alpha;sigma)), the realised gap depth
    phi_min: float                     # arccos(max_{I1}|kappa|), for the pass-band T_N bound
    TN_stop_bound: dict = field(default_factory=dict)  # {N: upper bound on max_{I0} T_N}, Prop. asymmetry(a)
    TN_pass_bound: dict = field(default_factory=dict)  # {N: lower bound on min_{I1} T_N}, Prop. asymmetry(b)
    accept: dict = field(default_factory=dict)         # {N: bool}, only if eps0/eps1 given


def certify(alphas: np.ndarray, J0: Sequence[Interval], J1: Sequence[Interval],
            sigma: Sequence[int], N_values: Sequence[int] = (),
            eps0: float | None = None, eps1: float | None = None) -> CertifyResult:
    """Exact certification (manuscript Sec. 6.3 "Certification" / Step 3 of
    the algorithm): report delta and mu_min to machine precision,
    independently of whatever grid design_direct's Phase 2 search used,
    then apply Proposition asymmetry's closed-form T_N bounds -- without
    ever evaluating T_N on a grid.
    """
    a = a_from_alphas(alphas)
    f = autocorr(a)

    delta = _extremize(f, J1, lambda th: q1_abs_sq(a, th) - 1.0, "max") if J1 else 0.0

    if J0:
        depth_min = min(
            _extremize(a, [(lo, hi)], lambda th, s=s: s * kappa_B(a, th), "min")
            for (lo, hi), s in zip(J0, sigma)
        )
        mu_min = float(np.arccosh(max(depth_min, 1.0)))
    else:
        mu_min = float("inf")

    kappa_abs_max = _extremize(a, J1, lambda th: np.abs(kappa_B(a, th)), "max") if J1 else 0.0
    phi_min = float(np.arccos(min(max(kappa_abs_max, -1.0), 1.0)))

    TN_stop_bound = {N: 1.0 / np.cosh(N * mu_min) ** 2 for N in N_values} if J0 else {}
    sin2phi = max(np.sin(phi_min) ** 2, 1e-300)
    TN_pass_bound = {N: 1.0 / (1.0 + delta / sin2phi) for N in N_values} if J1 else {}

    accept = {}
    if eps0 is not None and eps1 is not None:
        for N in N_values:
            ok_stop = TN_stop_bound.get(N, 0.0) <= eps0
            ok_pass = TN_pass_bound.get(N, 1.0) >= 1.0 - eps1
            accept[N] = bool(ok_stop and ok_pass)

    return CertifyResult(delta=delta, mu_min=mu_min, phi_min=phi_min,
                          TN_stop_bound=TN_stop_bound, TN_pass_bound=TN_pass_bound, accept=accept)
