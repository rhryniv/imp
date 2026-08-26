"""SLSQP problem generalised to multi-component I_1, I_0, per-component
sign sigma_i on (C). Reuses core.py's forward model unchanged. Decision
vector x=(alpha_0,...,alpha_{n-1},delta) as before.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from core import full_alphas, block_poly, autocorrelation, Q_from_f, Q_direct, kappa as kappa_fn

MU0 = 0.32303
COSH_MU0 = np.cosh(MU0)

GEOMETRIES = {
    "A": {
        "I1": [(0.0, np.pi / 4)],
        "I0": [(0.45 * np.pi, 0.60 * np.pi), (5 * np.pi / 6, np.pi)],
        "pi_in": "I0",
        "pi_component_index": 1,  # index into I0 list containing theta=pi
    },
    "B": {
        "I1": [(0.0, np.pi / 4), (0.85 * np.pi, np.pi)],
        "I0": [(0.45 * np.pi, 0.70 * np.pi)],
        "pi_in": "I1",
        "pi_component_index": 1,  # index into I1 list containing theta=pi
    },
}

OPT_N = 801
FINE_N = 16001


def make_grid(components, n_nodes):
    return [np.linspace(lo, hi, n_nodes) for lo, hi in components]


def concat_grid(components, n_nodes):
    return np.concatenate(make_grid(components, n_nodes))


def drop_theta0(theta_concat):
    """kappa(0)=1 identically (spec note) -- drop a theta=0 node from the
    (E) grid if present (first node of the whole concatenated I1 grid,
    when the first I1 component starts at 0, true for both geometries)."""
    if len(theta_concat) and np.isclose(theta_concat[0], 0.0):
        return theta_concat[1:]
    return theta_concat


def Q_on_grid(c, theta):
    f = autocorrelation(c)
    return Q_from_f(f, theta)


class Problem:
    def __init__(self, geom_name, n_nodes=OPT_N):
        g = GEOMETRIES[geom_name]
        self.name = geom_name
        self.I1 = g["I1"]
        self.I0 = g["I0"]
        self.m0 = len(self.I0)
        self.m1 = len(self.I1)
        self.G1 = concat_grid(self.I1, n_nodes)
        self.E1 = drop_theta0(self.G1)
        self.G0_list = make_grid(self.I0, n_nodes)

    def constraints_vec(self, x, n, L, sigma):
        alpha_first = x[:n]
        delta = x[n]
        alphas = full_alphas(alpha_first)
        c = block_poly(alphas)

        Q1 = Q_on_grid(c, self.G1)
        c_delta = delta - (Q1 - 1.0)

        c_C_parts = []
        for i, G0_i in enumerate(self.G0_list):
            kap0 = kappa_fn(c, L, G0_i)
            c_C_parts.append(sigma[i] * kap0 - COSH_MU0)

        kapE = kappa_fn(c, L, self.E1)
        c_E_hi = 1.0 - kapE
        c_E_lo = 1.0 + kapE

        return np.concatenate([c_delta] + c_C_parts + [c_E_hi, c_E_lo])

    def objective(self, x, n, L, sigma):
        return x[n]

    def objective_grad(self, x, n, L, sigma):
        g = np.zeros_like(x)
        g[n] = 1.0
        return g


ALPHA_BOUND = 20.0
DELTA_BOUND = (0.0, 10.0)


def solve_one(prob, x0, n, L, sigma, ftol=1e-14, maxiter=500):
    cons = [{"type": "ineq", "fun": prob.constraints_vec, "args": (n, L, sigma)}]
    bounds = [(-ALPHA_BOUND, ALPHA_BOUND)] * n + [DELTA_BOUND]
    res = minimize(prob.objective, x0, args=(n, L, sigma), jac=prob.objective_grad,
                    method="SLSQP", constraints=cons, bounds=bounds,
                    options={"ftol": ftol, "maxiter": maxiter})
    return res


def fine_diagnostics(prob, alpha_first, n, L, sigma):
    fine = Problem.__new__(Problem)
    fine.name = prob.name
    fine.I1, fine.I0 = prob.I1, prob.I0
    fine.m0, fine.m1 = prob.m0, prob.m1
    fine.G1 = concat_grid(prob.I1, FINE_N)
    fine.E1 = drop_theta0(fine.G1)
    fine.G0_list = make_grid(prob.I0, FINE_N)

    alphas = full_alphas(alpha_first)
    c = block_poly(alphas)

    Q1_fine = Q_on_grid(c, fine.G1)
    delta_fine = float(np.max(Q1_fine - 1.0))

    # min_Q over [0,pi] (union of I1, I0, and the gaps between)
    all_lo = min(lo for lo, hi in fine.I1 + fine.I0)
    all_hi = max(hi for lo, hi in fine.I1 + fine.I0)
    full_grid = np.linspace(0.0, np.pi, FINE_N)
    Q_full = Q_on_grid(c, full_grid)
    min_Q = float(np.min(np.concatenate([Q_full, Q1_fine])))

    kappa_min_per_C = []
    beta_per_active_C = []
    n_active_C_per_component = []
    for i, G0_i in enumerate(fine.G0_list):
        kap0 = kappa_fn(c, L, G0_i)
        signed = sigma[i] * kap0
        kappa_min_per_C.append(float(np.min(signed)))
        # active points of (C)_i: signed - cosh(mu0) ~ 0
        slack = signed - COSH_MU0
        active = np.abs(slack) < 1e-9 * max(1.0, COSH_MU0)
        n_active_C_per_component.append(int(np.sum(active)))
        Q_at_active = Q_on_grid(c, G0_i[active]) if np.any(active) else np.array([])
        beta_per_active_C.append([float(q - 1.0) for q in Q_at_active])

    kapE_fine = kappa_fn(c, L, fine.E1)
    max_abs_kappa_I1 = float(np.max(np.abs(kapE_fine)))

    # active points of (B) on I1: (Q1-1) close to delta_fine
    active_B = np.abs((Q1_fine - 1.0) - delta_fine) < 1e-9 * max(1.0, delta_fine)
    n_active_B_per_component = []
    idx = 0
    for lo, hi in prob.I1:
        seg = np.linspace(lo, hi, FINE_N)
        seg_active = np.abs((Q_on_grid(c, seg) - 1.0) - delta_fine) < 1e-9 * max(1.0, delta_fine)
        n_active_B_per_component.append(int(np.sum(seg_active)))

    kappa_min_overall = min(kappa_min_per_C) if kappa_min_per_C else None
    feasible = (kappa_min_overall is not None and kappa_min_overall >= COSH_MU0 - 1e-9
                and max_abs_kappa_I1 <= 1.0 + 1e-9 and min_Q >= 1.0 - 1e-12)

    theta_check = fine.G1[::500]
    q_f = Q_on_grid(c, theta_check)
    q_d = Q_direct(c, theta_check)
    cross_check_max_diff = float(np.max(np.abs(q_f - q_d)))

    return {"delta_fine": delta_fine, "kappa_min_per_C": kappa_min_per_C,
            "max_abs_kappa_I1": max_abs_kappa_I1, "min_Q": min_Q,
            "n_active_B_per_component": n_active_B_per_component,
            "n_active_C_per_component": n_active_C_per_component,
            "beta_per_active_C": beta_per_active_C,
            "feasible_fine": feasible, "cross_check_max_diff": cross_check_max_diff}


# ------------------------------------------------------------- start generators

def random_start(n, rng, sigma_std=0.35):
    return rng.normal(0.0, sigma_std, size=n)


def antisym_start(n, rng, sigma_std=0.35):
    assert n % 2 == 1
    n_free = (n + 1) // 2
    v = rng.normal(0.0, sigma_std, size=n_free)
    full = np.zeros(n + 1)
    for j, val in enumerate(v):
        full[j] = val
        full[n - j] = -val
    return full[:n]


def sym_start(n, rng, sigma_std=0.35):
    """alpha_j = alpha_{n-j} (symmetric, not antisymmetric). Since the
    matching condition forces sum=0, and a fully symmetric vector with n
    odd has a genuine free midpoint (index n/2 is not an integer for n
    odd... wait n odd -> n+1 even -> no fixed midpoint either, same
    pairing structure as antisym but with a PLUS sign): alpha_{n-j}=alpha_j
    for all pairs (j,n-j), j=0,...,(n-1)/2, free values drawn from the
    same N(0,0.35^2) distribution. NOTE this does NOT generally satisfy
    sum=0 (sum = 2*sum(free values) unless those happen to cancel), so we
    then re-impose matching the way the rest of the code does: alpha_n is
    determined as -sum(alpha_first) regardless, which BREAKS exact
    symmetry unless the free values already summed appropriately -- this
    is used only as a raw starting point for SLSQP, which does not need
    to start feasible or exactly symmetric; the symmetry is what P2 checks
    for at the OPTIMUM, not at the start."""
    assert n % 2 == 1
    n_free = (n + 1) // 2
    v = rng.normal(0.0, sigma_std, size=n_free)
    full = np.zeros(n + 1)
    for j, val in enumerate(v):
        full[j] = val
        full[n - j] = val
    return full[:n]


def embed_continuation(alpha_first_small, n_small, n_target):
    assert n_target == n_small + 2
    full_small = full_alphas(alpha_first_small)
    full_target = np.concatenate([[0.0], full_small, [0.0]])
    return full_target[:n_target]


def init_delta(prob, alpha_first, n, L, sigma):
    alphas = full_alphas(alpha_first)
    c = block_poly(alphas)
    Q1 = Q_on_grid(c, prob.G1)
    return float(np.max(Q1 - 1.0))


def build_starts(geom_name, n, rng, prev_same_n=None, prev_n_minus_2=None,
                  n_random=200, n_antisym=50, n_sym=0):
    starts = []
    for _ in range(n_random):
        starts.append((random_start(n, rng), "random"))
    for _ in range(n_antisym):
        starts.append((antisym_start(n, rng), "antisym"))
    for _ in range(n_sym):
        starts.append((sym_start(n, rng), "sym"))
    if prev_same_n is not None:
        starts.append((np.array(prev_same_n), "continuation"))
    if prev_n_minus_2 is not None:
        starts.append((embed_continuation(np.array(prev_n_minus_2), n - 2, n), "continuation"))
    return starts


def solve_case(geom_name, n, d, sigma, rng, prev_same_n=None, prev_n_minus_2=None,
               n_random=200, n_antisym=50, n_sym=0):
    prob = Problem(geom_name)
    L = n + d
    starts = build_starts(geom_name, n, rng, prev_same_n, prev_n_minus_2, n_random, n_antisym, n_sym)
    t0 = time.time()
    best = None
    n_converged = 0
    for alpha_first0, stype in starts:
        d0 = init_delta(prob, alpha_first0, n, L, sigma)
        x0 = np.concatenate([alpha_first0, [d0]])
        res = solve_one(prob, x0, n, L, sigma)
        if not res.success:
            continue
        alpha_first = res.x[:n].copy()
        diag = fine_diagnostics(prob, alpha_first, n, L, sigma)
        if not diag["feasible_fine"]:
            continue
        n_converged += 1
        delta_fine = diag["delta_fine"]
        if best is None or delta_fine < best["delta_fine"]:
            best = {"alpha_first": alpha_first, "delta_fine": delta_fine,
                    "start_type": stype, "iters": int(res.nit), "diag": diag}
    wall = time.time() - t0
    return best, n_converged, len(starts), wall
