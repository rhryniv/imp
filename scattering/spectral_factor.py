"""Fejer-Riesz spectral factorization.

Given real coefficients c_0,...,c_n with
    G(theta) = c_0 + 2 * sum_{l=1}^n c_l cos(l*theta)  >= 0   for all theta,
find real a_0,...,a_n such that

    | sum_{m=0}^n a_m e^{i m theta} |^2 = G(theta)   for all theta,

with the polynomial z -> sum_m a_m z^m having all its roots *inside* the
closed unit disc (the canonical / "minimum-phase" spectral factor).  This
is exactly the normalisation required by the layer-stripping theorem
(Thm 4.6 in the paper): p_1(w) = sum_j a_{n-j} w^j must be zero-free in the
closed unit disc |w|<=1, and p_1(w) = w^n * q~_1(1/w), so p_1 zero-free in
|w|<=1  <=>  q~_1(z) zero-free in |z|>=1  <=>  all roots of q~_1 lie
strictly inside |z|<1.

Two cases are handled:
  * G(theta) > 0 for all real theta (generic case; this is what happens
    for constraint (A), since the SDP enforces G >= 1): no roots of the
    associated degree-2n polynomial lie on |z|=1, and the factorization is
    a straightforward inside/outside split of its roots.
  * G(theta) >= 0 with possible double roots on |z|=1 (this happens for
    F = G - 1 = |q_2|^2, which is allowed to touch zero at "closed gaps"):
    handled by clustering near-unit-circle roots and assigning half of
    each cluster's multiplicity to the inside factor.
"""
from __future__ import annotations

import numpy as np


def _autocorr_to_full_poly(c: np.ndarray) -> np.ndarray:
    """c (len n+1) -> coefficients (len 2n+1, increasing powers) of z^n * G(theta)|_{z=e^{i theta}}."""
    n = len(c) - 1
    coeffs = np.zeros(2 * n + 1)
    coeffs[n] = c[0]
    for l in range(1, n + 1):
        coeffs[n - l] += c[l]
        coeffs[n + l] += c[l]
    return coeffs


def fejer_riesz_min_phase(c: np.ndarray, unit_circle_tol: float = 1e-6) -> np.ndarray:
    """Return a (len n+1), the minimum-phase spectral factor of G with autocorrelation c."""
    c = np.asarray(c, dtype=float)
    n = len(c) - 1
    if n == 0:
        return np.array([np.sqrt(max(c[0], 0.0))])

    coeffs = _autocorr_to_full_poly(c)   # increasing powers, length 2n+1
    roots = np.roots(coeffs[::-1])       # numpy.roots wants highest power first
    mags = np.abs(roots)

    on_circle = np.abs(mags - 1.0) <= unit_circle_tol
    inside = (~on_circle) & (mags < 1.0)
    outside = (~on_circle) & (mags >= 1.0)

    chosen = list(roots[inside])
    n_needed = n - len(chosen)

    if np.any(on_circle):
        chosen += _half_of_unit_circle_roots(roots[on_circle], n_needed)
    elif n_needed > 0:
        # Should not happen for a strictly generic case, but guard anyway:
        # take the smallest-magnitude remaining roots (numerically inside-ish).
        rest = roots[outside]
        order = np.argsort(np.abs(rest))
        chosen += list(rest[order[:n_needed]])

    chosen = np.array(chosen[:n])
    poly = np.poly(chosen)[::-1]   # increasing powers, monic (leading coeff of z^n is 1)
    poly = poly.real               # roots come in conjugate pairs -> coefficients are real

    scale_sq = c[0] / np.sum(poly ** 2)
    if scale_sq < 0:
        raise RuntimeError("negative scale in spectral factorization; G may not be nonnegative")
    a = np.sqrt(scale_sq) * poly
    return a


def _half_of_unit_circle_roots(unit_roots: np.ndarray, n_needed: int, cluster_tol: float = 1e-4):
    """Cluster near-unit-circle roots by angle and keep half the multiplicity of each cluster.

    G real & >=0 forces roots on |z|=1 to occur with even multiplicity, in
    complex-conjugate pairs (or as a real double root at z=+-1). Clustering
    handles the numerical fact that numpy.roots resolves an exact double
    root as two nearby-but-not-identical roots.
    """
    angles = np.angle(unit_roots)
    order = np.argsort(angles)
    angles_sorted = angles[order]
    roots_sorted = unit_roots[order]

    clusters = []
    cur = [roots_sorted[0]]
    for r, ang in zip(roots_sorted[1:], angles_sorted[1:]):
        if abs(ang - np.angle(cur[-1])) <= cluster_tol:
            cur.append(r)
        else:
            clusters.append(cur)
            cur = [r]
    clusters.append(cur)
    # wrap-around: first and last cluster might be the same near +-pi
    if len(clusters) > 1 and abs((angles_sorted[0] + 2 * np.pi) - np.angle(clusters[-1][-1])) <= cluster_tol:
        clusters[0] = clusters[-1] + clusters[0]
        clusters.pop()

    chosen = []
    for cl in clusters:
        keep = len(cl) // 2
        chosen += list(cl[:keep])
    if len(chosen) != n_needed:
        # fall back: just take n_needed of them (angle-sorted) -- degenerate/edge case
        all_pts = [r for cl in clusters for r in cl]
        chosen = all_pts[:n_needed]
    return chosen


def evaluate_G_from_a(a: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """|sum_m a_m e^{i m theta}|^2, for validating a spectral factor."""
    a = np.asarray(a, dtype=float)
    m = np.arange(len(a))
    z = np.exp(1j * np.outer(theta, m))
    q = z @ a
    return np.abs(q) ** 2
