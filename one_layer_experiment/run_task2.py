"""Task 2 -- illustration table at baseline t=pi/4, u=3pi/4.

Built from transfer_matrix.py's matrix machinery only (double precision;
grid convergence checked by halving the step, per the brief).
"""
from __future__ import annotations

import numpy as np

from transfer_matrix import one_layer_alphas, kappa_B, T_N, U_cheb2

T, U = np.pi / 4, 3 * np.pi / 4
L = 2
S_VALUES = [0.25, 0.5, 1, 2, 4]
N_TABLE = [2, 8, 16]


def alpha_of_s(s):
    return np.arcsinh(np.sqrt(s))


def field_arrays(s, grid):
    alpha = alpha_of_s(s)
    alphas = one_layer_alphas(alpha)
    kap = np.empty_like(grid)
    q2abs2 = np.empty_like(grid)
    for i, vartheta in enumerate(grid):
        k = vartheta / 2
        kb, q1, q2 = kappa_B(alphas, k, L)
        kap[i] = kb.real
        q2abs2[i] = abs(q2) ** 2
    return kap, q2abs2


def T_env(kap, q2abs2):
    out = np.full_like(kap, np.nan)
    mask = np.abs(kap) <= 1.0
    phi = np.arccos(np.clip(kap[mask], -1, 1))
    sin2phi = np.sin(phi) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        out[mask] = 1.0 / (1.0 + q2abs2[mask] / sin2phi)
    return out


def compute_row(s, n_grid):
    grid_I0 = np.linspace(U, np.pi, n_grid)
    grid_I1 = np.linspace(0.0, T, n_grid)
    kap0, q20 = field_arrays(s, grid_I0)
    kap1, q21 = field_arrays(s, grid_I1)

    mu_arr = np.arccosh(np.abs(kap0))
    mu_min = np.min(mu_arr)

    row = {"s": s, "mu_min": mu_min}
    for N in N_TABLE:
        TN0 = T_N(q20, kap0, N)
        TN1 = T_N(q21, kap1, N)
        row[f"max_I0_T{N}"] = np.max(TN0)
        row[f"min_I1_T{N}"] = np.min(TN1)

    Tenv1 = T_env(kap1, q21)
    row["min_I1_Tenv"] = np.nanmin(Tenv1)
    return row


def run():
    rows = []
    for s in S_VALUES:
        r1 = compute_row(s, 4001)
        r2 = compute_row(s, 8001)  # halved step, for convergence check
        max_rel_diff = max(abs(r1[k] - r2[k]) for k in r1 if k != "s")
        rows.append(r2)
        print(f"s={s:4.2f}  mu_min={r2['mu_min']:.6f}  "
              f"min_I1_T2={r2['min_I1_T2']:.6f} min_I1_T8={r2['min_I1_T8']:.6f} min_I1_T16={r2['min_I1_T16']:.6f}  "
              f"max_I0_T2={r2['max_I0_T2']:.3e} max_I0_T8={r2['max_I0_T8']:.3e} max_I0_T16={r2['max_I0_T16']:.3e}  "
              f"min_I1_Tenv={r2['min_I1_Tenv']:.6f}  grid_convergence_diff={max_rel_diff:.2e}")
    return rows


LATEX_HEADER = r"""\begin{tabular}{c|c|ccc|ccc|c}
\hline
$s$ & $\mu_{\min}$ & \multicolumn{3}{c|}{$\max_{I_0} T_N$} & \multicolumn{3}{c|}{$\min_{I_1} T_N$} & $\min_{I_1} T_{\mathrm{env}}$ \\
    &              & $N=2$ & $N=8$ & $N=16$ & $N=2$ & $N=8$ & $N=16$ & \\
\hline
"""


def to_latex(rows):
    lines = [LATEX_HEADER]
    for r in rows:
        lines.append(
            f"{r['s']:.2f} & {r['mu_min']:.4f} & "
            f"{r['max_I0_T2']:.2e} & {r['max_I0_T8']:.2e} & {r['max_I0_T16']:.2e} & "
            f"{r['min_I1_T2']:.4f} & {r['min_I1_T8']:.4f} & {r['min_I1_T16']:.4f} & "
            f"{r['min_I1_Tenv']:.4f} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


if __name__ == "__main__":
    rows = run()
    tex = to_latex(rows)
    with open("task2_table.tex", "w") as f:
        f.write(tex)
    print("\n" + tex)
