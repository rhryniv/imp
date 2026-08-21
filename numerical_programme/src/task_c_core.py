"""Task C core: fast double-precision NLP machinery (forward recursion,
Q, kappa_B) for the direct log-contrast optimization. mpmath is not used
here -- this is a numerical search, not a precision-critical closed-form
step; the winning point from each search is re-verified independently
(see task_c.py) before being trusted.
"""
from __future__ import annotations

import numpy as np


def forward_recursion_np(alphas):
    """alphas: 1D array [alpha_0..alpha_n]. Returns (p1, p2) as complex128
    coefficient arrays, low degree first, length n+1."""
    a0 = alphas[0]
    p1 = np.array([np.cosh(a0)], dtype=complex)
    p2 = np.array([np.sinh(a0)], dtype=complex)
    for aj in alphas[1:]:
        ca, sa = np.cosh(aj), np.sinh(aj)
        p2_shift = np.concatenate([[0.0], p2])  # w*p2
        p1_pad = np.concatenate([p1, [0.0]])
        new_p1 = ca * p1_pad + sa * p2_shift
        new_p2 = sa * p1_pad + ca * p2_shift
        p1, p2 = new_p1, new_p2
    return p1, p2


def poly_eval_np(coeffs, w):
    """coeffs low-degree first; w: array of complex points."""
    res = np.zeros_like(w, dtype=complex)
    for c in coeffs[::-1]:
        res = res * w + c
    return res


def Q_and_kappa(alphas, L, vartheta_grid):
    """alphas: [alpha_0..alpha_n]. Returns (Q, kappa_B) arrays over vartheta_grid."""
    p1, p2 = forward_recursion_np(alphas)
    w = np.exp(-1j * vartheta_grid)
    p1w = poly_eval_np(p1, w)
    Q = np.abs(p1w) ** 2
    kappa = np.real(np.exp(1j * L * vartheta_grid / 2) * p1w)
    return Q, kappa


def full_alphas(alpha_rest):
    """alpha_rest = [alpha_1..alpha_n]; alpha_0 = -sum."""
    a0 = -np.sum(alpha_rest)
    return np.concatenate([[a0], alpha_rest])
