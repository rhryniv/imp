"""mpmath (arbitrary precision) mirror of transfer_matrix.py, for the
high-precision cross-checks in Tasks 1, 4, 6. Same definitions, no
closed forms used here either -- built purely from Section 1."""
from __future__ import annotations

import mpmath as mp


def C_matrix_mp(alpha):
    ca, sa = mp.cosh(alpha), mp.sinh(alpha)
    return mp.matrix([[ca, sa], [sa, ca]])


def E_matrix_mp(phi):
    return mp.matrix([[mp.e ** (1j * phi), 0], [0, mp.e ** (-1j * phi)]])


def M_matrix_mp(alpha, kx):
    return E_matrix_mp(-kx) * C_matrix_mp(alpha) * E_matrix_mp(kx)


def block_M_mp(alphas, k, h=1):
    n = len(alphas) - 1
    M = mp.eye(2)
    for j in range(n, -1, -1):
        xj = j * h
        M = M * M_matrix_mp(alphas[j], k * xj)
    return M


def q1_q2_mp(alphas, k, h=1):
    M = block_M_mp(alphas, k, h)
    return M[0, 0], M[1, 0], M


def kappa_B_mp(alphas, k, L, h=1):
    q1, q2, M = q1_q2_mp(alphas, k, h)
    P = E_matrix_mp(k * L * h) * M
    return mp.mpf(1) / 2 * (P[0, 0] + P[1, 1]), q1, q2


def U_cheb2_mp(m, x):
    if m == -1:
        return mp.mpf(0)
    if m == 0:
        return mp.mpf(1)
    Um2, Um1 = mp.mpf(1), 2 * x
    if m == 1:
        return Um1
    for _ in range(2, m + 1):
        Um2, Um1 = Um1, 2 * x * Um1 - Um2
    return Um1


def T_N_mp(q2_abs2, kappa, N):
    U = U_cheb2_mp(N - 1, kappa)
    return 1 / (1 + q2_abs2 * U ** 2)


def one_layer_alphas_mp(alpha):
    return [alpha, -alpha]
