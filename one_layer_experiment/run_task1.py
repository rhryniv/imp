"""Task 1 -- independent implementation and cross-check.

Standing instruction obeyed: kappa_B, q1, q2, T_N are built ONLY from
transfer_matrix_mp.py's matrix products (Section 1 definitions); the
closed forms (closed_forms.py) are compared against, never used to
build anything.
"""
from __future__ import annotations

import mpmath as mp

from transfer_matrix_mp import one_layer_alphas_mp, kappa_B_mp, U_cheb2_mp, T_N_mp
from closed_forms import kappa_B_closed_mp, Q_closed_mp

mp.mp.dps = 30

S_VALUES = [mp.mpf("0.25"), mp.mpf("0.5"), mp.mpf(1), mp.mpf(2), mp.mpf(4)]
N_VALUES = [1, 2, 4, 8, 16]
L = 2
GRID_N = 400  # per s; mpmath is slow, keep this modest but fine enough to see max discrepancy


def alpha_of_s(s):
    return mp.asinh(mp.sqrt(s))


def run():
    report = []
    for s in S_VALUES:
        alpha = alpha_of_s(s)
        alphas = one_layer_alphas_mp(alpha)

        max_kappa_diff = mp.mpf(0)
        max_Q_diff = mp.mpf(0)
        max_su11_diff = mp.mpf(0)
        max_q1poly_diff = mp.mpf(0)

        grid = [mp.pi * i / GRID_N for i in range(1, GRID_N + 1)]  # (0, pi], vartheta=0 checked separately
        for vartheta in grid:
            k = vartheta / 2
            kap, q1, q2 = kappa_B_mp(alphas, k, L)
            kap_closed = kappa_B_closed_mp(vartheta, s)
            max_kappa_diff = max(max_kappa_diff, abs(kap.real - kap_closed))
            max_kappa_diff = max(max_kappa_diff, abs(kap.imag))  # should be exactly real

            Q = abs(q1) ** 2
            Q_closed = Q_closed_mp(vartheta, s)
            max_Q_diff = max(max_Q_diff, abs(Q - Q_closed))

            su11 = abs(q1) ** 2 - abs(q2) ** 2
            max_su11_diff = max(max_su11_diff, abs(su11 - 1))

            w = mp.e ** (-1j * vartheta)
            q1_poly = (1 + s) - s * w
            max_q1poly_diff = max(max_q1poly_diff, abs(q1 - q1_poly))

        # vartheta = 0 exactly
        kap0, q10, q20 = kappa_B_mp(alphas, mp.mpf(0), L)
        kappa0_err = abs(kap0.real - 1) + abs(kap0.imag)
        c0_plus_c1 = q10.real  # should be exactly 1 (q1(0) real)
        c0_plus_c1_err = abs(c0_plus_c1 - 1) + abs(q10.imag)

        TN0_errs = []
        for N in N_VALUES:
            TN0 = T_N_mp(abs(q20) ** 2, kap0.real, N)
            TN0_errs.append(abs(TN0 - 1))
        max_TN0_err = max(TN0_errs)

        # (D): c_0 = 1+s, c_1 = -s directly, via the polynomial-fit check above
        # (already captured in max_q1poly_diff)

        row = {
            "s": s, "max_kappa_diff": max_kappa_diff, "max_Q_diff": max_Q_diff,
            "max_su11_diff": max_su11_diff, "max_q1poly_diff": max_q1poly_diff,
            "kappa0_err": kappa0_err, "c0_plus_c1_err": c0_plus_c1_err,
            "max_TN0_err": max_TN0_err,
        }
        report.append(row)
        print(f"s={float(s):5.2f}  max|kappaB-closed|={mp.nstr(max_kappa_diff,4)}  "
              f"max|Q-closed|={mp.nstr(max_Q_diff,4)}  max||q1|^2-|q2|^2-1|={mp.nstr(max_su11_diff,4)}  "
              f"max|q1-poly|={mp.nstr(max_q1poly_diff,4)}  "
              f"kappaB(0)err={mp.nstr(kappa0_err,4)}  c0+c1 err={mp.nstr(c0_plus_c1_err,4)}  "
              f"max|T_N(0)-1|={mp.nstr(max_TN0_err,4)}")
    return report


if __name__ == "__main__":
    run()
