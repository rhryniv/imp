"""Core forward model: alphas -> c (block polynomial coeffs) -> Q, kappa.
Pure numpy, coefficient recursion only (no symbolic algebra), per spec
Section 1.
"""
from __future__ import annotations

import numpy as np


def full_alphas(alpha_first: np.ndarray) -> np.ndarray:
    """alpha_first = (alpha_0,...,alpha_{n-1}); alpha_n := -sum(alpha_first)
    (the matching condition sum_j alpha_j = 0)."""
    alpha_first = np.asarray(alpha_first, dtype=float)
    an = -np.sum(alpha_first)
    return np.concatenate([alpha_first, [an]])


def block_poly(alphas: np.ndarray) -> np.ndarray:
    """c = (c_0,...,c_n), coefficients of p_1(w)=sum_j c_j w^j, via the
    real coefficient recursion of Sec 1.2. alphas has length n+1."""
    alphas = np.asarray(alphas, dtype=float)
    p1 = np.array([np.cosh(alphas[0])])
    p2 = np.array([np.sinh(alphas[0])])
    for aj in alphas[1:]:
        ch, sh = np.cosh(aj), np.sinh(aj)
        p1_pad = np.concatenate([p1, [0.0]])   # p1^(j-1), degree bumped
        w_p2 = np.concatenate([[0.0], p2])      # w * p2^(j-1)
        p1, p2 = ch * p1_pad + sh * w_p2, sh * p1_pad + ch * w_p2
    return p1


def autocorrelation(c: np.ndarray) -> np.ndarray:
    """f_m = sum_{l=0}^{n-m} c_l c_{l+m}, m=0,...,n."""
    n = len(c) - 1
    f = np.zeros(n + 1)
    for m in range(n + 1):
        f[m] = np.dot(c[: n + 1 - m], c[m : n + 1])
    return f


def Q_from_f(f: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Q(theta) = f_0 + 2*sum_{m=1}^n f_m cos(m theta)."""
    q = np.full_like(theta, f[0], dtype=float)
    for m in range(1, len(f)):
        q += 2.0 * f[m] * np.cos(m * theta)
    return q


def Q_direct(c: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """|p1(e^{-i theta})|^2, direct complex evaluation -- used only as an
    independent cross-check against Q_from_f (spec Sec 7, conditioning
    trap: must agree to 1e-10)."""
    w = np.exp(-1j * theta)
    n = len(c) - 1
    p1w = np.zeros_like(w, dtype=complex)
    for j in range(n, -1, -1):
        p1w = p1w * w + c[j]
    return np.abs(p1w) ** 2


def kappa(c: np.ndarray, L: float, theta: np.ndarray) -> np.ndarray:
    """kappa(theta) = sum_j c_j cos((L-2j) theta / 2)."""
    val = np.zeros_like(theta, dtype=float)
    for j, cj in enumerate(c):
        val += cj * np.cos((L - 2 * j) * theta / 2.0)
    return val


def rho_from_alphas(alpha_first: np.ndarray) -> np.ndarray:
    """rho_i = exp(sum_{j<i} alpha_j), i=1,...,n (reporting only).
    sum_{j<i} alpha_j for j=0,...,i-1 -> the i-th partial cumsum of the
    full (n+1)-length alpha vector, i=1,...,n (the n-th, i.e. full sum
    minus alpha_n, equals -alpha_n since the total is 0)."""
    alphas = full_alphas(alpha_first)
    cum = np.cumsum(alphas)  # cum[i-1] = sum_{j=0}^{i-1} alpha_j = sum_{j<i} alpha_j
    return np.exp(cum[:-1])  # rho_1,...,rho_n (drop cum[n]=0, which is not a layer)


def A_minus1(alpha_first: np.ndarray) -> float:
    """A_{-1} := sum_j (-1)^j alpha_j, over the full (n+1)-length alpha
    vector (the endpoint-theta=pi diagnostic of the multi-geometry spec)."""
    alphas = full_alphas(alpha_first)
    signs = np.array([(-1) ** j for j in range(len(alphas))], dtype=float)
    return float(np.dot(signs, alphas))
