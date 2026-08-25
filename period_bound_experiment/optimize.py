"""SLSQP optimisation problem per spec Sections 1.4/2. Decision vector
x = (alpha_0,...,alpha_{n-1}, delta) in R^{n+1}. Objective: minimise
delta. Constraints (B) epigraph, (C) stop-band, (E) pass-band |kappa|<=1
with theta=0 dropped from the E grid (spec's degeneracy note).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from core import full_alphas, block_poly, autocorrelation, Q_from_f, kappa as kappa_fn

I1 = (0.0, np.pi / 4)
I0 = (5 * np.pi / 6, np.pi)
MU0 = 0.32303
COSH_MU0 = np.cosh(MU0)
G1_N = 801
G0_N = 801

G1 = np.linspace(I1[0], I1[1], G1_N)
G0 = np.linspace(I0[0], I0[1], G0_N)
E_GRID = G1[1:]  # drop theta=0: kappa(0)=1 identically, constrains nothing

FINE_N = 16001
G1_FINE = np.linspace(I1[0], I1[1], FINE_N)
G0_FINE = np.linspace(I0[0], I0[1], FINE_N)
E_FINE = G1_FINE[1:]


def Q_on_grid(c, theta):
    f = autocorrelation(c)
    return Q_from_f(f, theta)


def constraints_vec(x, n, L, sigma):
    alpha_first = x[:n]
    delta = x[n]
    alphas = full_alphas(alpha_first)
    c = block_poly(alphas)

    Q1 = Q_on_grid(c, G1)
    c_delta = delta - (Q1 - 1.0)

    kap0 = kappa_fn(c, L, G0)
    c_C = sigma * kap0 - COSH_MU0

    kapE = kappa_fn(c, L, E_GRID)
    c_E_hi = 1.0 - kapE
    c_E_lo = 1.0 + kapE

    return np.concatenate([c_delta, c_C, c_E_hi, c_E_lo])


def objective(x, n, L, sigma):
    return x[n]


def objective_grad(x, n, L, sigma):
    g = np.zeros_like(x)
    g[n] = 1.0
    return g


# Bounds are not specified in the brief; these are added purely as a
# numerical-stability guard (not a scientific restriction -- the true
# optima are all O(0.1-0.5), and cosh/sinh only start overflowing float64
# past |alpha|~700). Without them, a badly-conditioned finite-difference
# SLSQP step at n=5 was observed to run alpha out to the point of
# cosh/sinh overflow, producing NaNs that stalled convergence and made
# n=5 cases ~25x slower than n=1. Disclosed in config.json / summary.md.
ALPHA_BOUND = 20.0
DELTA_BOUND = (0.0, 10.0)


def solve_one(x0, n, L, sigma, ftol=1e-14, maxiter=500):
    cons = [{"type": "ineq", "fun": constraints_vec, "args": (n, L, sigma)}]
    bounds = [(-ALPHA_BOUND, ALPHA_BOUND)] * n + [DELTA_BOUND]
    res = minimize(objective, x0, args=(n, L, sigma), jac=objective_grad,
                    method="SLSQP", constraints=cons, bounds=bounds,
                    options={"ftol": ftol, "maxiter": maxiter})
    return res


FEAS_TOL = 1e-9


def is_feasible(x, n, L, sigma, tol=FEAS_TOL):
    cv = constraints_vec(x, n, L, sigma)
    return bool(np.all(cv >= -tol)), float(cv.min())


def fine_diagnostics(alpha_first, n, L, sigma):
    """Sec 3 fine-grid (16001-node) verification: kappa_min, kappa_max,
    delta_fine, min_Q, plus the f-vs-direct-Q cross-check (Sec 7 trap)."""
    alphas = full_alphas(alpha_first)
    c = block_poly(alphas)

    Q1_fine = Q_on_grid(c, G1_FINE)
    delta_fine = float(np.max(Q1_fine - 1.0))

    Q0_fine = Q_on_grid(c, G0_FINE)
    Q_full_fine = np.concatenate([Q1_fine, Q0_fine,
                                   Q_on_grid(c, np.linspace(I1[1], I0[0], 2001))])
    min_Q = float(np.min(Q_full_fine))
    Q_J = float(np.max(Q0_fine))

    kap0_fine = kappa_fn(c, L, G0_FINE)
    kappa_min = float(np.min(sigma * kap0_fine))

    kapE_fine = kappa_fn(c, L, E_FINE)
    kappa_max = float(np.max(np.abs(kapE_fine)))

    # cross-check: Q via autocorrelation f_m vs direct |p1(e^{-i theta})|^2
    from core import Q_direct
    theta_check = G1_FINE[::500]
    q_f = Q_on_grid(c, theta_check)
    q_d = Q_direct(c, theta_check)
    cross_check_max_diff = float(np.max(np.abs(q_f - q_d)))

    feasible = (kappa_min >= COSH_MU0 - 1e-9) and (kappa_max <= 1.0 + 1e-9) and (min_Q >= 1.0 - 1e-12)
    return {"delta_fine": delta_fine, "kappa_min": kappa_min, "kappa_max": kappa_max,
            "min_Q": min_Q, "Q_J": Q_J, "feasible_fine": feasible,
            "cross_check_max_diff": cross_check_max_diff}


# ------------------------------------------------------------- start generators

def random_start(n, rng, sigma_std=0.35):
    alpha_first = rng.normal(0.0, sigma_std, size=n)
    return alpha_first


def antisym_start(n, rng, sigma_std=0.35):
    """Draw the (n+1)//2 free values of a fully antisymmetric length-(n+1)
    vector (alpha_j = -alpha_{n-j}), n odd so there is no fixed midpoint;
    return alpha_first = the first n entries (alpha_n is then automatically
    -sum(alpha_first), which for an antisymmetric vector equals -alpha_0,
    consistent with the antisymmetry by construction -- verified in dev)."""
    assert n % 2 == 1
    n_free = (n + 1) // 2
    v = rng.normal(0.0, sigma_std, size=n_free)
    full = np.zeros(n + 1)
    for j, val in enumerate(v):
        full[j] = val
        full[n - j] = -val
    return full[:n]


KNOWN_BLOCK_N3 = np.array([0.16228, -0.39891, 0.39891, -0.16228])  # full alpha, n=3


def init_delta(alpha_first, n, L, sigma):
    alphas = full_alphas(alpha_first)
    c = block_poly(alphas)
    Q1 = Q_on_grid(c, G1)
    return float(np.max(Q1 - 1.0))


def embed_continuation(alpha_first_small, n_small, n_target):
    """Embed an (n_small)-layer solution into an (n_target)-layer one by
    padding with a zero-contrast (trivial) layer at each end, n_target =
    n_small+2. This is the natural 'grow the block by two no-op layers'
    embedding; not specified exactly in the brief, so disclosed here and
    in the run's config/summary."""
    assert n_target == n_small + 2
    full_small = full_alphas(alpha_first_small)  # length n_small+1
    full_target = np.concatenate([[0.0], full_small, [0.0]])  # length n_target+1
    return full_target[:n_target]


def build_starts(n, sigma, rng, prev_same_n=None, prev_n_minus_2=None,
                  n_random=200, n_antisym=50, include_seed_ex=True):
    starts = []  # list of (alpha_first, start_type)
    for _ in range(n_random):
        starts.append((random_start(n, rng), "random"))
    for _ in range(n_antisym):
        starts.append((antisym_start(n, rng), "antisym"))
    if n == 3 and include_seed_ex:
        starts.append((KNOWN_BLOCK_N3[:3].copy(), "seed_ex"))
    if prev_same_n is not None:
        starts.append((prev_same_n.copy(), "continuation"))
    if prev_n_minus_2 is not None:
        starts.append((embed_continuation(prev_n_minus_2, n - 2, n), "continuation"))
    return starts


def solve_case(n, d, sigma, rng, prev_same_n=None, prev_n_minus_2=None,
               n_random=200, n_antisym=50, include_seed_ex=True, verbose=False):
    """Discard criterion per spec Sec 2: fail to converge, OR terminate
    infeasible at tolerance 1e-9 measured on the FINE (16001-node) grid
    of Sec 3 (not the optimisation grid). Selection among survivors and
    the reported delta both use delta_fine (Sec 3: 'this, not the
    solver's delta, is the reported value')."""
    L = n + d
    starts = build_starts(n, sigma, rng, prev_same_n, prev_n_minus_2, n_random, n_antisym,
                           include_seed_ex=include_seed_ex)
    t0 = time.time()
    best = None  # dict
    n_converged = 0
    for alpha_first0, stype in starts:
        d0 = init_delta(alpha_first0, n, L, sigma)
        x0 = np.concatenate([alpha_first0, [d0]])
        res = solve_one(x0, n, L, sigma)
        if not res.success:
            continue
        alpha_first = res.x[:n].copy()
        diag = fine_diagnostics(alpha_first, n, L, sigma)
        if not diag["feasible_fine"]:
            continue
        n_converged += 1
        delta_fine = diag["delta_fine"]
        if best is None or delta_fine < best["delta_fine"]:
            best = {"alpha_first": alpha_first, "delta_fine": delta_fine,
                    "delta_solver": float(res.x[n]), "start_type": stype,
                    "iters": int(res.nit), "diag": diag}
    wall = time.time() - t0
    return best, n_converged, len(starts), wall
