"""Forward problem: alpha's -> a -> kappa_B -> T_N.

alphas = (alpha_0, ..., alpha_n) are the log-impedance contrasts of an
n-layer block (eq. 3.1 in the paper). Everything here is pure evaluation:
no optimization, no factorization (that's inverse.py and sdp_design.py).
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as C


def forward_reconstruct(alphas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build (p1, p2) from alphas via the recursion of eqs. 4.5-4.6:

        p_1^(0) = cosh(alpha_0),   p_2^(0) = sinh(alpha_0)
        p_1^(j) = cosh(alpha_j) p_1^(j-1)     + sinh(alpha_j) [w p_2^(j-1)]
        p_2^(j) = sinh(alpha_j) p_1^(j-1)     + cosh(alpha_j) [w p_2^(j-1)]

    p1, p2 are returned as coefficient arrays of increasing powers of w,
    length n+1 (zero-padded to a common length along the way). This is
    the forward transfer-matrix recursion -- always well defined for any
    real alphas, and it automatically satisfies |p1|^2-|p2|^2=1 on |w|=1
    and p1 zero-free in the closed unit disc (Prop 4.4), i.e. it can only
    ever produce a *realizable* q~_1.
    """
    alphas = np.asarray(alphas, dtype=float)
    P1, P2 = np.array([np.cosh(alphas[0])]), np.array([np.sinh(alphas[0])])
    for alpha_j in alphas[1:]:
        ch, sh = np.cosh(alpha_j), np.sinh(alpha_j)
        P1_padded = np.concatenate([P1, [0.0]])         # p_1^(j-1), degree bumped (no shift)
        wP2 = np.concatenate([[0.0], P2])                # w * p_2^(j-1)  (shift up by one)
        P1_new = ch * P1_padded + sh * wP2
        P2_new = sh * P1_padded + ch * wP2
        P1, P2 = P1_new, P2_new
    return P1, P2


def a_from_alphas(alphas: np.ndarray) -> np.ndarray:
    """alphas -> a = q~_1 coefficients. a_m = p1[n-m] (eq. after 5.6: a_m = c_{n-m})."""
    p1, _ = forward_reconstruct(alphas)
    return p1[::-1].copy()


def kappa_B(a: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """kappa_B(theta) = Re(sum_m a_m e^{i m theta}) = sum_m a_m cos(m theta)
    = Chebyshev evaluation of a at x = cos(theta) (eq. 5.7-5.8)."""
    return C.chebval(np.cos(theta), a)


def q1_abs_sq(a: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """|q~_1(theta)|^2 = G(theta)."""
    a = np.asarray(a, dtype=float)
    m = np.arange(len(a))
    q = np.exp(1j * np.outer(theta, m)) @ a
    return np.abs(q) ** 2


def q2_abs_sq(a: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """|q_2(theta)|^2 = |q~_1(theta)|^2 - 1 (eq. 5.11), clipped to >= 0."""
    return np.clip(q1_abs_sq(a, theta) - 1.0, 0.0, None)


def _chebyshev_U(m: int, x: np.ndarray) -> np.ndarray:
    """Chebyshev polynomial of the second kind, U_m(x)."""
    x = np.asarray(x, dtype=float)
    if m < 0:
        return np.zeros_like(x)
    U_km1 = np.ones_like(x)          # U_0
    if m == 0:
        return U_km1
    U_k = 2 * x                       # U_1
    for _ in range(2, m + 1):
        U_km1, U_k = U_k, 2 * x * U_k - U_km1
    return U_k


def transmission_TN(a: np.ndarray, theta: np.ndarray, N: int) -> np.ndarray:
    """T_N(theta) = 1 / (1 + |q_2|^2 * U_{N-1}(kappa_B)^2)   (eq. 4.13)."""
    kap = kappa_B(a, theta)
    q2sq = q2_abs_sq(a, theta)
    U = _chebyshev_U(N - 1, kap)
    return 1.0 / (1.0 + q2sq * U ** 2)


def plot_filter(a, N_values, theta_range=(1e-3, np.pi - 1e-3), I0=None, I1=None,
                 n_grid=4000, savepath=None):
    """Figure with panels for kappa_B(theta), |q~_1(theta)|^2, and T_N(theta)
    (one curve per N in N_values). Shades I0 (stop, red) / I1 (pass, green)
    if given (each a list of (lo, hi) pairs in theta).

    Returns the matplotlib Figure; also saves to savepath if given.
    """
    import matplotlib.pyplot as plt

    theta = np.linspace(theta_range[0], theta_range[1], n_grid)
    kap = kappa_B(a, theta)
    G = q1_abs_sq(a, theta)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(theta, kap, color="black")
    axes[0].axhline(1.0, color="gray", lw=0.7, ls="--")
    axes[0].axhline(-1.0, color="gray", lw=0.7, ls="--")
    axes[0].set_ylabel(r"$\kappa_B(\theta)$")

    axes[1].plot(theta, G, color="black")
    axes[1].axhline(1.0, color="gray", lw=0.7, ls="--")
    axes[1].set_ylabel(r"$|\tilde q_1(\theta)|^2$")

    for N in N_values:
        axes[2].plot(theta, transmission_TN(a, theta, N), label=f"N={N}")
    axes[2].set_ylabel(r"$T_N(\theta)$")
    axes[2].set_xlabel(r"$\theta = 2kh$")
    axes[2].legend()

    for ax in axes:
        for lo, hi in (I0 or []):
            ax.axvspan(lo, hi, color="red", alpha=0.15)
        for lo, hi in (I1 or []):
            ax.axvspan(lo, hi, color="green", alpha=0.15)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig
