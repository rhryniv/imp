"""Task 5 -- harder instance: t=0.4*pi, u=0.6*pi (pass/stop close
together). Repeats Task 2's table and Task 3's three claims."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from transfer_matrix import one_layer_alphas, kappa_B, T_N

T, U = 0.4 * np.pi, 0.6 * np.pi
L = 2
S_VALUES = [0.25, 0.5, 1, 2, 4]
N_TABLE = [2, 8, 16]
MU0_EXCLUSION = 0.5


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


def T_env_min(s, t_hi, n_grid=100001):
    grid = np.linspace(0.0, t_hi, n_grid)
    kap, q2abs2 = field_arrays(s, grid)
    phi = np.arccos(np.clip(kap, -1, 1))
    sin2phi = np.sin(phi) ** 2
    ratio = np.empty_like(kap)
    ratio[1:] = q2abs2[1:] / sin2phi[1:]
    ratio[0] = s
    return np.min(1.0 / (1.0 + ratio))


def compute_row(s, n_grid):
    grid_I0 = np.linspace(U, np.pi, n_grid)
    grid_I1 = np.linspace(0.0, T, n_grid)
    kap0, q20 = field_arrays(s, grid_I0)
    kap1, q21 = field_arrays(s, grid_I1)
    mu_min = np.min(np.arccosh(np.abs(kap0)))
    row = {"s": s, "mu_min": mu_min}
    for N in N_TABLE:
        row[f"max_I0_T{N}"] = np.max(T_N(q20, kap0, N))
        row[f"min_I1_T{N}"] = np.min(T_N(q21, kap1, N))
    row["min_I1_Tenv"] = T_env_min(s, T)
    return row


def task2_repeat():
    print(f"=== Task 5: table at t={T:.4f}={T/np.pi:.2f}pi, u={U:.4f}={U/np.pi:.2f}pi ===")
    for s in S_VALUES:
        r = compute_row(s, 4001)
        print(f"s={s:4.2f}  mu_min={r['mu_min']:.6f}  "
              f"min_I1_T2={r['min_I1_T2']:.6f} min_I1_T8={r['min_I1_T8']:.6f} min_I1_T16={r['min_I1_T16']:.6f}  "
              f"max_I0_T2={r['max_I0_T2']:.3e} max_I0_T8={r['max_I0_T8']:.3e} max_I0_T16={r['max_I0_T16']:.3e}  "
              f"min_I1_Tenv={r['min_I1_Tenv']:.6f}")


def task3_repeat():
    print("\n=== Task 5: 3(1) non-monotonicity, s=1 ===")
    for N in range(1, 17):
        grid = np.linspace(0.0, T, 20001)
        kap, q2abs2 = field_arrays(1.0, grid)
        m = np.min(T_N(q2abs2, kap, N))
        print(f"  N={N:2d}  min_I1_T_N = {m:.6f}")
    env = T_env_min(1.0, T)
    print(f"  envelope = {env:.6f}")

    print("\n=== Task 5: 3(3) exclusion, mu0=0.5 ===")

    def eps1(s):
        return 1.0 - T_env_min(s, T)

    from closed_forms import s_minus
    sm = s_minus(MU0_EXCLUSION, U)
    res = minimize_scalar(eps1, bounds=(sm, sm * 5), method="bounded", options={"xatol": 1e-10})
    print(f"  s_-(mu0=0.5) = {sm:.6f}")
    print(f"  minimizing s = {res.x:.6f}  smallest achievable eps_1 = {res.fun:.6f}")
    print(f"  (C) active: s_opt/s_- = {res.x/sm:.6f}")
    print(f"  --> {'FAILS (eps_1 too large for a usable filter)' if res.fun > 0.4 else 'marginal/OK'}")


if __name__ == "__main__":
    task2_repeat()
    task3_repeat()
