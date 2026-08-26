"""SLSQP problem for Geometry C (two pass components, one stop, pi in
I0, sigma forced). Reuses the multi-component constraint machinery from
the previous experiment (generic over any number of I1/I0 components);
here I0 always has exactly one component and sigma is a 1-tuple.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from core import full_alphas, block_poly, autocorrelation, Q_from_f, Q_direct, kappa as kappa_fn

MU0 = 0.32303
COSH_MU0 = np.cosh(MU0)

GEOMETRIES = {
    "C": {
        "I1": [(0.0, 0.20 * np.pi), (0.40 * np.pi, 0.55 * np.pi)],
        "I0": [(0.78 * np.pi, np.pi)],
    },
    "G62": {  # Sec 6.2 gate geometry
        "I1": [(0.0, np.pi / 4)],
        "I0": [(5 * np.pi / 6, np.pi)],
    },
}

OPT_N = 801
FINE_N = 16001


def make_grid(components, n_nodes):
    return [np.linspace(lo, hi, n_nodes) for lo, hi in components]


def concat_grid(components, n_nodes):
    return np.concatenate(make_grid(components, n_nodes))


def drop_theta0(theta_concat):
    if len(theta_concat) and np.isclose(theta_concat[0], 0.0):
        return theta_concat[1:]
    return theta_concat


def Q_on_grid(c, theta):
    return Q_from_f(autocorrelation(c), theta)


class Problem:
    def __init__(self, geom_name, n_nodes=OPT_N):
        g = GEOMETRIES[geom_name]
        self.name = geom_name
        self.I1 = g["I1"]
        self.I0 = g["I0"]
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


def cluster(mask):
    """Number of contiguous True-runs, and their (start,end) index spans."""
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


def fine_diagnostics(prob, alpha_first, n, L, sigma):
    I1_fine_grids = make_grid(prob.I1, FINE_N)
    I0_fine_grids = make_grid(prob.I0, FINE_N)
    G1_fine = np.concatenate(I1_fine_grids)

    alphas = full_alphas(alpha_first)
    c = block_poly(alphas)

    Q1_fine = Q_on_grid(c, G1_fine)
    delta_fine = float(np.max(Q1_fine - 1.0))

    # per-I1-component delta and clustered active-point count (P5, P1's p)
    delta_per_component = []
    p_per_component = []
    for seg in I1_fine_grids:
        qseg = Q_on_grid(c, seg)
        delta_per_component.append(float(np.max(qseg - 1.0)))
    for seg in I1_fine_grids:
        qseg = Q_on_grid(c, seg)
        mask = np.abs((qseg - 1.0) - delta_fine) < 1e-9 * max(1.0, delta_fine)
        cnt, _ = cluster(mask)
        p_per_component.append(cnt)
    p_total = sum(p_per_component)

    # min_Q over [0,pi]
    full_grid = np.linspace(0.0, np.pi, FINE_N)
    Q_full = Q_on_grid(c, full_grid)
    min_Q = float(np.min(np.concatenate([Q_full, Q1_fine])))

    # (C): single I0 component here
    G0_fine = I0_fine_grids[0]
    kap0 = kappa_fn(c, L, G0_fine)
    signed = sigma[0] * kap0
    kappa_min_I0 = float(np.min(signed))
    slack_C = signed - COSH_MU0
    mask_C = np.abs(slack_C) < 1e-9 * max(1.0, COSH_MU0)
    a_count, a_spans = cluster(mask_C)
    Qseg0 = Q_on_grid(c, G0_fine)
    betas = [float(Qseg0[(s0 + s1) // 2] - 1.0) for s0, s1 in a_spans]

    # (E): active points on I1, EXCLUDING theta=0 (kappa(0)=1 identically --
    # spec Sec 5 item 5: must exclude it, "the previous run ... was
    # therefore uninformative")
    e_count_total = 0
    max_abs_kappa_I1_excl0 = 0.0
    for lo, hi in prob.I1:
        seg = np.linspace(lo, hi, FINE_N)
        if np.isclose(lo, 0.0):
            seg = seg[1:]  # drop theta=0
        if len(seg) == 0:
            continue
        kap_seg = kappa_fn(c, L, seg)
        max_abs_kappa_I1_excl0 = max(max_abs_kappa_I1_excl0, float(np.max(np.abs(kap_seg))))
        mask_E = np.abs(np.abs(kap_seg) - 1.0) < 1e-9
        cnt, _ = cluster(mask_E)
        e_count_total += cnt

    feasible = (kappa_min_I0 >= COSH_MU0 - 1e-9 and max_abs_kappa_I1_excl0 <= 1.0 + 1e-9
                and min_Q >= 1.0 - 1e-12)

    theta_check = G1_fine[::500]
    q_f = Q_on_grid(c, theta_check)
    q_d = Q_direct(c, theta_check)
    cross_check_max_diff = float(np.max(np.abs(q_f - q_d)))

    return {"delta_fine": delta_fine, "delta_per_component": delta_per_component,
            "p_per_component": p_per_component, "p_total": p_total,
            "kappa_min_I0": kappa_min_I0, "a_count": a_count, "betas": betas,
            "e_count": e_count_total, "max_abs_kappa_I1_excl0": max_abs_kappa_I1_excl0,
            "min_Q": min_Q, "feasible_fine": feasible,
            "cross_check_max_diff": cross_check_max_diff}


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


def build_starts(n, rng, prev_same_n=None, prev_n_minus_2=None, n_random=200, n_antisym=50):
    starts = []
    for _ in range(n_random):
        starts.append((random_start(n, rng), "random"))
    for _ in range(n_antisym):
        starts.append((antisym_start(n, rng), "antisym"))
    if prev_same_n is not None:
        starts.append((np.array(prev_same_n), "continuation_d-2"))
    if prev_n_minus_2 is not None:
        starts.append((embed_continuation(np.array(prev_n_minus_2), n - 2, n), "continuation_n-2"))
    return starts


def solve_case(geom_name, n, d, sigma, rng, prev_same_n=None, prev_n_minus_2=None,
               n_random=200, n_antisym=50):
    prob = Problem(geom_name)
    L = n + d
    starts = build_starts(n, rng, prev_same_n, prev_n_minus_2, n_random, n_antisym)
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
