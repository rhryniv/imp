"""Post-hoc re-analysis of the 6 feasible optima: proper clustered
active-point counts (the raw per-grid-point count used during the sweep
overcounts near-flat equioscillation peaks as many "active" points --
e.g. n_active_B=[45] for A_n5_d1 is a grid artifact, not 45 distinct
equioscillation points), plus P4 (T_N vs eps_0 formula).
"""
from __future__ import annotations

import json

import numpy as np

from core import full_alphas, block_poly, autocorrelation, Q_from_f, kappa as kappa_fn
from optimize_multi import GEOMETRIES, MU0, COSH_MU0, make_grid, drop_theta0

FINE_N = 16001


def Q_on_grid(c, theta):
    return Q_from_f(autocorrelation(c), theta)


def count_clusters(mask):
    """Number of contiguous True-runs in a boolean array (each run =
    one equioscillation point, however many adjacent fine-grid nodes it
    spans due to local flatness)."""
    if not np.any(mask):
        return 0, []
    edges = np.diff(mask.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    ends = list(np.where(edges == -1)[0])
    if mask[-1]:
        ends = ends + [len(mask) - 1]
    return len(starts), list(zip(starts, ends))


def analyze_case(geom_name, alpha_first, n, L, sigma):
    g = GEOMETRIES[geom_name]
    I1, I0 = g["I1"], g["I0"]
    alphas = full_alphas(np.array(alpha_first))
    c = block_poly(alphas)

    G1_fine = np.concatenate(make_grid(I1, FINE_N))
    delta_fine = float(np.max(Q_on_grid(c, G1_fine) - 1.0))

    # per-I1-component clustered active count for (B)
    n_active_B = []
    for lo, hi in I1:
        seg = np.linspace(lo, hi, FINE_N)
        qseg = Q_on_grid(c, seg)
        mask = np.abs((qseg - 1.0) - delta_fine) < 1e-9 * max(1.0, delta_fine)
        cnt, spans = count_clusters(mask)
        n_active_B.append(cnt)

    # per-I0-component clustered active count for (C)
    n_active_C = []
    kappa_min_per_C = []
    beta_per_active_C = []
    for i, (lo, hi) in enumerate(I0):
        seg = np.linspace(lo, hi, FINE_N)
        kap = kappa_fn(c, L, seg)
        signed = sigma[i] * kap
        kappa_min_per_C.append(float(np.min(signed)))
        mask = np.abs(signed - COSH_MU0) < 1e-9 * max(1.0, COSH_MU0)
        cnt, spans = count_clusters(mask)
        n_active_C.append(cnt)
        betas = []
        for s0, s1 in spans:
            mid = (s0 + s1) // 2
            betas.append(float(Q_on_grid(c, np.array([seg[mid]]))[0] - 1.0))
        beta_per_active_C.append(betas)

    return {"delta_fine": delta_fine, "n_active_B": n_active_B, "n_active_C": n_active_C,
            "kappa_min_per_C": kappa_min_per_C, "beta_per_active_C": beta_per_active_C}


def U_m(x, m):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    lt = np.abs(x) < 1
    eq1 = np.isclose(x, 1.0)
    eqm1 = np.isclose(x, -1.0)
    gt = (~lt) & (~eq1) & (~eqm1)
    u = np.arccos(np.clip(x[lt], -1, 1))
    out[lt] = np.sin((m + 1) * u) / np.sin(u) if m >= 0 else 0.0
    out[eq1] = m + 1
    out[eqm1] = ((-1) ** m) * (m + 1)
    if np.any(gt):
        xg = x[gt]
        s = np.sign(xg)
        uu = np.arccosh(np.abs(xg))
        val = np.sinh((m + 1) * uu) / np.sinh(uu) if m >= 0 else 0.0
        out[gt] = np.where(s > 0, val, ((-1) ** m) * val)
    return out


def T_N(Q, kappa, N):
    if N == 0:
        return np.ones_like(Q)
    return 1.0 / (1.0 + (Q - 1.0) * U_m(kappa, N - 1) ** 2)


def p4_table(geom_name, alpha_first, n, L, sigma, Ns=(1, 2, 5, 9, 20)):
    g = GEOMETRIES[geom_name]
    I0 = g["I0"]
    alphas = full_alphas(np.array(alpha_first))
    c = block_poly(alphas)

    rows = []
    for i, (lo, hi) in enumerate(I0):
        seg = np.linspace(lo, hi, FINE_N)
        Qseg = Q_on_grid(c, seg)
        kap = kappa_fn(c, L, seg)
        signed = sigma[i] * kap
        active_mask = np.abs(signed - COSH_MU0) < 1e-9 * max(1.0, COSH_MU0)
        betas_i = [float(Qseg[j] - 1.0) for j in np.where(active_mask)[0]]
        # cluster to distinct betas (avoid double counting a flat run)
        cnt, spans = count_clusters(active_mask)
        betas_distinct = [float(Qseg[(s0 + s1) // 2] - 1.0) for s0, s1 in spans]

        for N in Ns:
            Tn = T_N(Qseg, kap, N)
            max_TN_direct = float(np.max(Tn))
            # eps_0(N) formula per active point, take the max (spec Sec 5 P4)
            eps0_candidates = []
            for beta in betas_distinct:
                denom = 1.0 + beta * (np.sinh(N * MU0) ** 2) / (np.sinh(MU0) ** 2)
                eps0_candidates.append(1.0 / denom)
            eps0_formula = max(eps0_candidates) if eps0_candidates else None
            rows.append({"I0_component": i, "N": N, "max_TN_direct": max_TN_direct,
                         "eps0_formula": eps0_formula,
                         "betas_used": betas_distinct})
    return rows


if __name__ == "__main__":
    with open("results_multi_checkpoint.json") as f:
        R = json.load(f)

    feas = {k: v for k, v in R.items() if v["feasible"]}
    reanalysis = {}
    for k, v in feas.items():
        n, L, sigma = v["n"], v["L"], v["sigma"]
        alpha_first = v["alpha"][:n]
        diag = analyze_case(v["geometry"], alpha_first, n, L, sigma)
        p4 = p4_table(v["geometry"], alpha_first, n, L, sigma)
        reanalysis[k] = {"diag": diag, "p4": p4}
        print(f"{k}: n_active_B(clustered)={diag['n_active_B']}  "
              f"n_active_C(clustered)={diag['n_active_C']}  delta={diag['delta_fine']:.6e}")

    with open("reanalysis.json", "w") as f:
        json.dump(reanalysis, f, indent=2, default=str)
    print("saved reanalysis.json")
