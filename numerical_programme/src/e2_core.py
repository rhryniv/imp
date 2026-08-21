"""E2 core: fast double-precision NLP machinery. alpha_n = -sum(alpha_0..
alpha_{n-1}) -- THIS brief eliminates the LAST index (alpha_n), not the
first, unlike the earlier worked-example brief. Supports multiple I_0
components (Spec C) with one sign per component.
"""
from __future__ import annotations

import numpy as np


def forward_recursion_np(alphas):
    a0 = alphas[0]
    p1 = np.array([np.cosh(a0)], dtype=complex)
    p2 = np.array([np.sinh(a0)], dtype=complex)
    for aj in alphas[1:]:
        ca, sa = np.cosh(aj), np.sinh(aj)
        p2_shift = np.concatenate([[0.0], p2])
        p1_pad = np.concatenate([p1, [0.0]])
        new_p1 = ca * p1_pad + sa * p2_shift
        new_p2 = sa * p1_pad + ca * p2_shift
        p1, p2 = new_p1, new_p2
    return p1, p2


def poly_eval_np(coeffs, w):
    res = np.zeros_like(w, dtype=complex)
    for c in coeffs[::-1]:
        res = res * w + c
    return res


def Q_and_kappa(alphas, L, theta_grid):
    p1, p2 = forward_recursion_np(alphas)
    w = np.exp(-1j * theta_grid)
    p1w = poly_eval_np(p1, w)
    Q = np.abs(p1w) ** 2
    kappa = np.real(np.exp(1j * L * theta_grid / 2) * p1w)
    return Q, kappa


def full_alphas(alpha_first):
    """alpha_first = [alpha_0..alpha_{n-1}]; alpha_n = -sum (LAST index
    eliminated, per this brief's E2 convention)."""
    an = -np.sum(alpha_first)
    return np.concatenate([alpha_first, [an]])


def assert_sanity(alphas, tol=1e-9):
    """The three cheap §1.1 sanity checks, active every call."""
    p1, p2 = forward_recursion_np(alphas)
    p1_at_1 = np.sum(p1).real
    assert abs(p1_at_1 - 1.0) < tol, f"p1(1)={p1_at_1} != 1"
    p1_at_0 = p1[0].real
    expected = np.prod(np.cosh(alphas))
    assert abs(p1_at_0 - expected) < tol, f"p1(0)={p1_at_0} != prod cosh={expected}"
    thetas = np.linspace(0, np.pi, 50)
    Q, _ = Q_and_kappa(alphas, 2, thetas)  # L irrelevant to Q
    w = np.exp(-1j * thetas)
    p2w = poly_eval_np(p2, w)
    su11 = Q - np.abs(p2w) ** 2
    assert np.max(np.abs(su11 - 1.0)) < 1e-6, f"|p1|^2-|p2|^2 != 1, max dev {np.max(np.abs(su11-1))}"
    return True
