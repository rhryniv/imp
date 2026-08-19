"""Double-precision transfer-matrix machinery, built ONLY from the
definitions in the brief's Section 1 -- no closed forms used anywhere in
this file. Used for Tasks 2, 3, 5 (tables) and as the double-precision
half of the Task 1/4/6 cross-checks (the high-precision half is in
transfer_matrix_mp.py).
"""
from __future__ import annotations

import numpy as np


def C_matrix(alpha):
    ca, sa = np.cosh(alpha), np.sinh(alpha)
    return np.array([[ca, sa], [sa, ca]], dtype=complex)


def E_matrix(phi):
    return np.array([[np.exp(1j * phi), 0.0], [0.0, np.exp(-1j * phi)]], dtype=complex)


def M_matrix(alpha, kx):
    return E_matrix(-kx) @ C_matrix(alpha) @ E_matrix(kx)


def block_M(alphas, k, h=1.0):
    """alphas = [alpha_0, ..., alpha_n] (n+1 contrasts for n internal
    layers). Interfaces at x_j = j*h, j=0..n. Product taken j=n..0 (M at
    the last interface leftmost), per the brief."""
    n = len(alphas) - 1
    M = np.eye(2, dtype=complex)
    for j in range(n, -1, -1):
        xj = j * h
        M = M @ M_matrix(alphas[j], k * xj)
    return M


def q1_q2(alphas, k, h=1.0):
    M = block_M(alphas, k, h)
    return M[0, 0], M[1, 0], M


def kappa_B(alphas, k, L, h=1.0):
    """kappa_B(k) = 0.5*trace(E(kLh) M_B(k))."""
    q1, q2, M = q1_q2(alphas, k, h)
    P = E_matrix(k * L * h) @ M
    return 0.5 * np.trace(P), q1, q2


def U_cheb2(m, x):
    """Chebyshev polynomial of the second kind U_m(x), via the standard
    3-term recursion (works for any real x, including |x|>1; elementwise
    for numpy arrays). U_{-1}=0, U_0=1."""
    x = np.asarray(x, dtype=float)
    if m == -1:
        return np.zeros_like(x)
    if m == 0:
        return np.ones_like(x)
    Um2 = np.ones_like(x)   # U_0
    Um1 = 2.0 * x           # U_1
    if m == 1:
        return Um1
    for _ in range(2, m + 1):
        Um2, Um1 = Um1, 2.0 * x * Um1 - Um2
    return Um1


def T_N(q2_abs2, kappa, N):
    """T_N(k) = 1/(1+|q2|^2 U_{N-1}(kappa_B)^2)."""
    U = U_cheb2(N - 1, kappa)
    return 1.0 / (1.0 + q2_abs2 * U ** 2)


def one_layer_alphas(alpha):
    """n=1 block: rho_1=e^alpha, background 1. alpha_0=log(rho_1/rho_0)=alpha,
    alpha_1=log(rho_2/rho_1)=log(1/rho_1)=-alpha."""
    return [alpha, -alpha]
