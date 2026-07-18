"""Layer stripping (Schur algorithm), Theorem 4.6: recover the interface
contrasts alpha_0,...,alpha_n (hence the physical layer impedances) from
the block coefficient vector a = (a_0,...,a_n) (i.e. tilde_q1).

p_1(w) = sum_j a_{n-j} w^j   (degree n, real, zero-free in the closed unit
disc -- guaranteed by construction: a is the *minimum-phase* spectral
factor of G, see spectral_factor.py's docstring for the w <-> z duality).

p_2(w) is any polynomial with |p_1|^2 - |p_2|^2 = 1 on |w|=1 (Remark 4.9:
any spectral factor is admissible); we again take the minimum-phase one
via Fejer-Riesz, of F(theta) = |tilde_q1(theta)|^2 - 1 = |q2(theta)|^2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral_factor import fejer_riesz_min_phase


def ensure_min_phase(a: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Project a onto the minimum-phase representative of the same G(theta)
    = |q~_1(theta)|^2, if it isn't already one.

    The design SDP (design_full.py) + local polish only constrain the
    *magnitude* |q~_1| (that's all constraints (A)/(B) see) plus the linear
    constraints (C)/(D) on a directly; nothing in that problem forces the
    roots of q~_1(z) = sum a_m z^m inside the unit disc, so the optimizer
    is free to (and in practice often does) return the max-phase or a
    mixed-phase branch instead of the physically-realizable min-phase one
    required by Theorem 4.6. All such branches share the same |q~_1|, so
    this reflects any outside root z0 to its reciprocal 1/conj(z0) (leaves
    G unchanged) and renormalizes so sum(a) = 1 (constraint D). NOTE this
    can change kappa_B = Re(q~_1), so constraint (C) must be re-checked
    (design_full.verify_design) on the returned vector.
    """
    a = np.asarray(a, dtype=float)
    n = len(a) - 1
    if n == 0:
        return a.copy()
    roots = np.roots(a[::-1])
    if np.all(np.abs(roots) < 1.0 + tol):
        return a.copy()
    c = np.array([np.sum(a[: n + 1 - l] * a[l:]) for l in range(n + 1)])
    a_mp = fejer_riesz_min_phase(c)
    return a_mp * (np.sum(a) / np.sum(a_mp))  # keep the same overall sign/normalization


def p1_from_a(a: np.ndarray) -> np.ndarray:
    """p_1(w) = sum_j a_{n-j} w^j, i.e. just the coefficient vector reversed."""
    return np.asarray(a, dtype=float)[::-1].copy()


def p2_from_a(a: np.ndarray) -> np.ndarray:
    """A spectral factor of F(theta) = |q~_1(theta)|^2 - 1 = |q2(theta)|^2
    (Remark 4.9: any spectral factor is admissible here, unlike p1)."""
    a = np.asarray(a, dtype=float)
    n = len(a) - 1
    c = np.array([np.sum(a[: n + 1 - l] * a[l:]) for l in range(n + 1)])  # autocorrelation of a
    c[0] -= 1.0
    return fejer_riesz_min_phase(c)


@dataclass
class LayerStack:
    alphas: np.ndarray            # alpha_0,...,alpha_n  (log-contrasts)
    impedances: np.ndarray        # p_0=1, p_1,...,p_n, p_{n+1} (should be ~1)
    residual_sum_alpha: float     # should be ~0 (matched background, eq. 4.4)
    reconstruction_error: float   # ||forward_recursion(alphas) - (p1,p2)||, sanity check
    a_used: np.ndarray            # the (possibly min-phase-projected) coefficient vector actually used
    was_reflected: bool           # True if a had to be projected to min phase (see ensure_min_phase)


def schur_strip(p1: np.ndarray, p2: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Theorem 4.6 recursion. p1, p2: real coeff vectors (increasing powers
    of w), length n+1, with |p1|^2-|p2|^2=1 on |w|=1 and p1 zero-free in
    the closed unit disc. Returns alphas = (alpha_0,...,alpha_n)."""
    P1 = np.asarray(p1, dtype=float).copy()
    P2 = np.asarray(p2, dtype=float).copy()
    n = len(P1) - 1
    if len(P2) != n + 1:
        raise ValueError("p1 and p2 must have the same length")
    alphas = np.zeros(n + 1)
    for j in range(n, 0, -1):
        ratio = P2[0] / P1[0]
        if abs(ratio) >= 1.0:
            raise RuntimeError(f"Schur parameter out of (-1,1) at step j={j}: {ratio}")
        alpha_j = np.arctanh(ratio)
        alphas[j] = alpha_j
        ch, sh = np.cosh(alpha_j), np.sinh(alpha_j)
        newP1 = ch * P1 - sh * P2       # degree <= j-1: newP1[j] ~ 0
        newP2 = ch * P2 - sh * P1       # newP2[0] ~ 0 by construction of alpha_j
        P1 = newP1[:j]
        P2 = newP2[1 : j + 1]
    alphas[0] = np.arctanh(P2[0] / P1[0])
    return alphas


def forward_reconstruct(alphas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of schur_strip: rebuild (p1, p2) from alphas (eq. 4.5-4.6), a
    consistency check on the peeling recursion."""
    P1, P2 = np.array([np.cosh(alphas[0])]), np.array([np.sinh(alphas[0])])
    for alpha_j in alphas[1:]:
        ch, sh = np.cosh(alpha_j), np.sinh(alpha_j)
        wP1, wP2 = np.concatenate([[0.0], P1]), np.concatenate([[0.0], P2])
        P1_new = ch * np.concatenate([P1, [0.0]]) + sh * wP2
        P2_new = sh * np.concatenate([P1, [0.0]]) + ch * wP2
        P1, P2 = P1_new, P2_new
    return P1, P2


def layers_from_a(a: np.ndarray) -> LayerStack:
    """Full pipeline: a -> (p1,p2) -> alphas -> physical impedances p_0..p_{n+1}.

    a is first projected to the minimum-phase representative of the same
    G = |q~_1|^2 if it isn't one already (see ensure_min_phase) -- required
    for Theorem 4.6's zero-free hypothesis on p1. If this reflection was
    needed, re-check constraint (C) on stack.a_used afterwards (e.g. via
    design_full.verify_design), since kappa_B = Re(a_used) can change.
    """
    a = np.asarray(a, dtype=float)
    a_mp = ensure_min_phase(a)
    was_reflected = not np.allclose(a_mp, a, atol=1e-9)

    p1 = p1_from_a(a_mp)
    p2 = p2_from_a(a_mp)
    alphas = schur_strip(p1, p2)

    impedances = np.empty(len(alphas) + 2)
    impedances[0] = 1.0
    for j, alpha_j in enumerate(alphas):
        impedances[j + 1] = impedances[j] * np.exp(alpha_j)

    p1_chk, p2_chk = forward_reconstruct(alphas)
    err = max(np.max(np.abs(p1_chk - p1)), np.max(np.abs(p2_chk - p2)))

    return LayerStack(
        alphas=alphas,
        impedances=impedances,
        residual_sum_alpha=float(np.sum(alphas)),
        reconstruction_error=float(err),
        a_used=a_mp,
        was_reflected=was_reflected,
    )
