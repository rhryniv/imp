"""Physical quantities derived from a single-block coefficient vector a_m:
the Floquet discriminant kappa_B, the off-diagonal q2, and the N-block
transmission probability T_N (eq. 4.13 / 5.1 in the paper)."""
from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as C


def kappa_B(a: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """kappa_B(theta) = Re(sum_m a_m e^{i m theta}) = sum_m a_m cos(m theta)."""
    x = np.cos(theta)
    return C.chebval(x, a)


def q2_from_a(a: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """|q2(theta)|, from |q~_1|^2 - 1 = |q2|^2 (eq. 5.11)."""
    from .spectral_factor import evaluate_G_from_a
    G = evaluate_G_from_a(a, theta)
    return np.sqrt(np.clip(G - 1.0, 0.0, None))


def _cheb2_eval(m: int, x: np.ndarray) -> np.ndarray:
    # U_0=1, U_1=2x, U_k = 2x U_{k-1} - U_{k-2}
    x = np.asarray(x, dtype=float)
    if m < 0:
        return np.zeros_like(x)
    Ukm1 = np.ones_like(x)   # U_0
    if m == 0:
        return Ukm1
    Uk = 2 * x            # U_1
    for _ in range(2, m + 1):
        Ukm1, Uk = Uk, 2 * x * Uk - Ukm1
    return Uk


def transmission_TN(a: np.ndarray, theta: np.ndarray, N: int) -> np.ndarray:
    """T_N(theta) = 1 / (1 + |q2|^2 * U_{N-1}(kappa_B)^2)."""
    kap = kappa_B(a, theta)
    q2 = q2_from_a(a, theta)
    U = _cheb2_eval(N - 1, kap)
    return 1.0 / (1.0 + (q2 ** 2) * (U ** 2))


def achieved_gap_depth(a: np.ndarray, J0, n_grid: int = 4000):
    """min_{theta in J0} arccosh(|kappa_B(theta)|), and the achieved sign pattern.

    Returns (mu_min, per_interval) where per_interval is a list of
    (alpha, beta, mu_j, sigma_j) with sigma_j = sign of kappa_B on that
    component (the constraint-(C) sign convention), or sigma_j=None /
    mu_j=0 if kappa_B does not stay outside [-1,1] on that component
    (i.e. constraint (C) genuinely fails there).
    """
    per_interval = []
    mus = []
    for alpha, beta in J0:
        th = np.linspace(alpha, beta, n_grid)
        kap = kappa_B(a, th)
        if np.all(kap >= 1.0):
            sigma = 1
            mu = np.arccosh(np.min(kap))
        elif np.all(kap <= -1.0):
            sigma = -1
            mu = np.arccosh(np.min(-kap))
        else:
            sigma = None
            mu = 0.0
        per_interval.append((alpha, beta, mu, sigma))
        mus.append(mu)
    return (min(mus) if mus else float("inf")), per_interval
