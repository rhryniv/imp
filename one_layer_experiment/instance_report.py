"""Worked instance: n=1, s=1, baseline t=pi/4, u=3pi/4 -- Tasks 1-2 of
the follow-up brief. Built from transfer_matrix.py's matrix machinery
(already validated against the closed forms to 30 digits in the earlier
run); closed_forms.py used only as a cross-check here, per the standing
instruction.
"""
from __future__ import annotations

import numpy as np

from transfer_matrix import one_layer_alphas, kappa_B, T_N
from closed_forms import kappa_B_closed, Q_closed

T, U = np.pi / 4, 3 * np.pi / 4
L = 2
S = 1.0
N_VALUES = [2, 8, 16]

EARLIER_RUN = {
    "mu_min": 1.5286,
    "max_I0_TN": {2: 6.24e-3, 8: 6.76e-11, 16: 1.62e-21},
    "min_I1_TN": {2: 0.4576, 8: 0.4372, 16: 0.4238},
    "min_I1_Tenv": 0.414214,
}


def alpha_of_s(s):
    return np.arcsinh(np.sqrt(s))


def field_arrays(s, grid):
    alpha = alpha_of_s(s)
    alphas = one_layer_alphas(alpha)
    kap = np.empty_like(grid)
    Q = np.empty_like(grid)
    q2abs2 = np.empty_like(grid)
    for i, vartheta in enumerate(grid):
        kb, q1, q2 = kappa_B(alphas, vartheta / 2, L)
        kap[i] = kb.real
        Q[i] = abs(q1) ** 2
        q2abs2[i] = abs(q2) ** 2
    return kap, Q, q2abs2


def T_env_min(s, t_hi, n_grid):
    grid = np.linspace(0.0, t_hi, n_grid)
    kap, Q, q2abs2 = field_arrays(s, grid)
    phi = np.arccos(np.clip(kap, -1, 1))
    sin2phi = np.sin(phi) ** 2
    ratio = np.empty_like(kap)
    ratio[1:] = q2abs2[1:] / sin2phi[1:]
    ratio[0] = s  # vartheta->0 limit, see earlier session's review
    return np.min(1.0 / (1.0 + ratio))


def compute(n_grid):
    grid_I0 = np.linspace(U, np.pi, n_grid)
    grid_I1 = np.linspace(0.0, T, n_grid)
    kap0, Q0, q20 = field_arrays(S, grid_I0)
    kap1, Q1, q21 = field_arrays(S, grid_I1)

    # cross-check matrix vs closed form on both grids
    kap0_closed = kappa_B_closed(grid_I0, S)
    kap1_closed = kappa_B_closed(grid_I1, S)
    Q0_closed = Q_closed(grid_I0, S)
    Q1_closed = Q_closed(grid_I1, S)
    max_kappa_diff = max(np.max(np.abs(kap0 - kap0_closed)), np.max(np.abs(kap1 - kap1_closed)))
    max_Q_diff = max(np.max(np.abs(Q0 - Q0_closed)), np.max(np.abs(Q1 - Q1_closed)))

    mu_min = np.min(np.arccosh(np.abs(kap0)))
    delta = np.max(Q1 - 1.0)

    row = {"mu_min": mu_min, "delta": delta, "max_kappa_diff": max_kappa_diff, "max_Q_diff": max_Q_diff}
    for N in N_VALUES:
        TN0 = T_N(q20, kap0, N)
        TN1 = T_N(q21, kap1, N)
        row[f"eps0_{N}"] = np.max(TN0)
        row[f"eps1_{N}"] = 1.0 - np.min(TN1)
        row[f"minI1_TN_{N}"] = np.min(TN1)  # for cross-check against earlier run's own units
    row["Tenv_floor"] = 1.0 - T_env_min(S, T, n_grid)
    row["min_I1_Tenv"] = T_env_min(S, T, n_grid)
    return row


def run():
    r1 = compute(4001)
    r2 = compute(8001)
    max_grid_diff = max(abs(r1[k] - r2[k]) for k in r1)
    r = r2

    alpha = alpha_of_s(S)
    rho1 = np.exp(alpha)

    print("=== Task 1 ===")
    print(f"alpha = {alpha:.6f}")
    print(f"rho_1 = {rho1:.6f}")
    print(f"mu_min = {r['mu_min']:.4g}")
    print(f"delta  = {r['delta']:.4g}")
    for N in N_VALUES:
        print(f"N={N:2d}  eps_0(N) = {r[f'eps0_{N}']:.4g}   eps_1(N) = {r[f'eps1_{N}']:.4g}   "
              f"(min_I1_T_N = {r[f'minI1_TN_{N}']:.4g})")
    print(f"1 - min_I1 T_env = {r['Tenv_floor']:.6g}  (min_I1_T_env = {r['min_I1_Tenv']:.6g})")
    print(f"\ngrid convergence (4001 vs 8001 pts), max diff over all reported quantities: {max_grid_diff:.2e}")
    print(f"matrix-vs-closed-form cross-check: max|kappaB diff|={r['max_kappa_diff']:.2e}  "
          f"max|Q diff|={r['max_Q_diff']:.2e}")

    print("\n--- cross-check against earlier run ---")
    print(f"mu_min: this run {r['mu_min']:.4f} vs earlier {EARLIER_RUN['mu_min']:.4f}  "
          f"diff={abs(r['mu_min']-EARLIER_RUN['mu_min']):.2e}")
    for N in N_VALUES:
        d0 = abs(r[f'eps0_{N}'] - EARLIER_RUN['max_I0_TN'][N])
        d1 = abs(r[f'minI1_TN_{N}'] - EARLIER_RUN['min_I1_TN'][N])
        print(f"N={N:2d}  max_I0_T_N diff={d0:.2e}   min_I1_T_N diff={d1:.2e}")
    print(f"min_I1_Tenv: this run {r['min_I1_Tenv']:.6f} vs earlier {EARLIER_RUN['min_I1_Tenv']:.6f}  "
          f"vs sqrt(2)-1={np.sqrt(2)-1:.6f}  diff={abs(r['min_I1_Tenv']-EARLIER_RUN['min_I1_Tenv']):.2e}")

    print("\n=== Task 2: surrogate bound T_N >= 1/(1+N^2 delta) ===")
    delta = r["delta"]
    for N in N_VALUES:
        surrogate_eps1 = 1.0 - 1.0 / (1.0 + N ** 2 * delta)
        true_eps1 = r[f"eps1_{N}"]
        print(f"N={N:2d}  surrogate eps_1 = {surrogate_eps1:.4g}   true eps_1(N) = {true_eps1:.4g}   "
              f"surrogate/true = {surrogate_eps1/true_eps1:.3g}")

    return r, alpha, rho1


LATEX_TABLE = r"""\begin{{tabular}}{{c|c|c}}
\hline
$N$ & $\epsilon_0(N)$ & $\epsilon_1(N)$ \\
\hline
2  & {e0_2} & {e1_2} \\
8  & {e0_8} & {e1_8} \\
16 & {e0_16} & {e1_16} \\
\hline
\end{{tabular}}
"""


def to_latex(r):
    return LATEX_TABLE.format(
        e0_2=f"{r['eps0_2']:.3g}", e1_2=f"{r['eps1_2']:.4g}",
        e0_8=f"{r['eps0_8']:.3g}", e1_8=f"{r['eps1_8']:.4g}",
        e0_16=f"{r['eps0_16']:.3g}", e1_16=f"{r['eps1_16']:.4g}",
    )


if __name__ == "__main__":
    r, alpha, rho1 = run()
    tex = to_latex(r)
    with open("instance_table.tex", "w") as f:
        f.write(tex)
    print("\n" + tex)
