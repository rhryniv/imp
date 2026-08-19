"""Task 3 -- the three claims the table is meant to support."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from transfer_matrix import one_layer_alphas, kappa_B, T_N
from closed_forms import s_minus

T, U = np.pi / 4, 3 * np.pi / 4
L = 2
MU0_EXCLUSION = 0.5  # confirmed with the user


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


def T_env_min(s, t_hi, n_grid=200001):
    grid = np.linspace(0.0, t_hi, n_grid)
    kap, q2abs2 = field_arrays(s, grid)
    # vartheta -> 0 limit of |q2|^2/sin^2(phi) is exactly s (see brief review); handle the
    # first grid point (vartheta=0 exactly) via this analytic limit instead of 0/0.
    phi = np.arccos(np.clip(kap, -1, 1))
    sin2phi = np.sin(phi) ** 2
    ratio = np.empty_like(kap)
    ratio[1:] = q2abs2[1:] / sin2phi[1:]
    ratio[0] = s
    Tenv = 1.0 / (1.0 + ratio)
    return np.min(Tenv)


def min_I1_TN(s, N, n_grid=20001):
    grid = np.linspace(0.0, T, n_grid)
    kap, q2abs2 = field_arrays(s, grid)
    return np.min(T_N(q2abs2, kap, N))


def part1_nonmonotonicity():
    print("=== 3(1) Non-monotonicity: min_I1 T_N, N=1..16, s=1 ===")
    rows = []
    for N in range(1, 17):
        m = min_I1_TN(1.0, N)
        rows.append((N, m))
        print(f"  N={N:2d}  min_I1_T_N = {m:.6f}")
    env = T_env_min(1.0, T)
    print(f"  envelope min_I1_T_env = {env:.6f}")
    diffs = [m - env for _, m in rows]
    print(f"  min_I1_T_N - envelope ranges from {min(diffs):.6f} to {max(diffs):.6f} "
          f"(sign changes: {sum(1 for i in range(len(diffs)-1) if diffs[i]*diffs[i+1]<0)})")
    return rows, env


def part2_asymmetry():
    print("\n=== 3(2) Asymmetry (from Task 2's table, N=8) ===")
    from run_task2 import S_VALUES, compute_row
    stopband, passband = [], []
    for s in S_VALUES:
        r = compute_row(s, 4001)
        stopband.append(r["max_I0_T8"])
        passband.append(r["min_I1_T8"])
    print(f"  stop-band max_I0_T8 range over s in {S_VALUES}: [{min(stopband):.3e}, {max(stopband):.3e}]  "
          f"({np.log10(max(stopband)/min(stopband)):.1f} orders of magnitude)")
    print(f"  pass-band min_I1_T8 range over the same s: [{min(passband):.4f}, {max(passband):.4f}]  "
          f"(spread {max(passband)-min(passband):.4f})")
    return stopband, passband


def part3_exclusion(mu0=MU0_EXCLUSION):
    print(f"\n=== 3(3) Exclusion, mu0={mu0} ===")
    sm = s_minus(mu0, U)

    def eps1(s):
        return 1.0 - T_env_min(s, T)

    # confirm s_-(mu0) is indeed the minimizer by scanning slightly above it too
    probe_s = [sm, sm * 1.001, sm * 1.01, sm * 1.1, sm * 2]
    for s in probe_s:
        print(f"  s={s:.6f} (s/s_-={s/sm:.4f})  eps_1={eps1(s):.6f}")

    res = minimize_scalar(eps1, bounds=(sm, sm * 3), method="bounded",
                           options={"xatol": 1e-10})
    s_opt = res.x
    eps1_opt = res.fun
    print(f"  s_-(mu0={mu0}) = {sm:.6f}")
    print(f"  minimizing s (scipy, over [s_-, 3 s_-]) = {s_opt:.6f}  eps_1 = {eps1_opt:.6f}")
    print(f"  (C) active at the minimizer: s_opt/s_- = {s_opt/sm:.8f} (1.0 = active/boundary)")
    return sm, s_opt, eps1_opt


if __name__ == "__main__":
    part1_nonmonotonicity()
    part2_asymmetry()
    part3_exclusion()
