"""Stage 2, spec Sec. 3.2 ("Two phases"): the two-phase direct optimiser,
implemented literally against codespec6.4.md -- a deliberate rewrite,
not a reuse of sdp_design.py's earlier `design_direct`, because the two
now differ in several load-bearing ways the Authority rule (spec Sec. 0)
requires flagging rather than silently reconciling:

  - Phase 1 runs ALL 2^m0 sign patterns to convergence (no early exit).
    A SECOND, user-approved deviation from spec Sec. 3.2's literal "the
    pattern with the largest value is passed to Phase 2": Phase 2 is run
    for EVERY Phase-1-feasible pattern, not just the single largest-margin
    one, and the best (min delta) result across all of them is kept --
    confirmed directly (n=7 multiband case) that the largest-margin
    pattern is not always the one that actually flattens well
    (sigma=(-1,-1), kappa_min=1.926, gave delta=57.9, while the discarded
    sigma=(-1,1), kappa_min=1.00125, gave the correct delta=0.0277671...,
    exactly reproducing the earlier design_direct's own reason for the
    same choice). Cost is modest: m_0 is typically 1-3 in the manuscript's
    own instances, so at most 2^m_0<=8 Phase 2 solves. kappa_min_by_sigma
    is still recorded for every pattern regardless, as the direct emission
    source (spec Sec. 6) and so any future regression stays visible.
  - Phase 2's start pool is exactly the three the spec lists (the
    Phase-1 point, zero-padded lower-degree embeddings, small random
    draws), with NO screening for (C)-feasibility of any start -- the
    spec is explicit that padded starts are typically (C)-infeasible by
    construction (L shifts from 2n to 2n+2) and would all be wrongly
    discarded by a screen. An optional, off-by-default fourth member
    (the magnitude-SDP factorisation) can be added purely to answer the
    manuscript's own open question about that warm start (tracked via
    start_origin).

A third, user-approved deviation: Phase 1 is a RAMPED-TARGET feasibility
search (ramp the required depth from 0 to mu0 in stages, warm-started
stage to stage), not the literal unconstrained "max_{alpha,t} t"
epigraph of spec Sec. 3.2. The literal form has no interior optimum --
more contrast always helps kappa_min -- so it drives every free alpha_j
to exactly the box boundary regardless of how loose the box is;
confirmed empirically on the n=7 multiband case (kappa_min~2.2e9, alpha
components pinned at the box edge, and the resulting Phase 2 seed giving
delta=57.9 instead of the known-good 0.0277671370191698). This was
surfaced to the user with that evidence before implementing, per the
Authority rule, rather than silently substituted.

Throughout, the optimisation variable is gamma_free = tanh(alpha_free)
(bounded, so the SQP always has a well-posed feasible region no matter
how large alpha itself grows -- a code-authority decision, spec Sec. 8).
The chain rule through arctanh, d(alpha_free_j)/d(gamma_free_j) =
1/(1-gamma_free_j^2), is applied explicitly to every constraint Jacobian
built here; sdp_design.py's own version of this (the now-superseded
design_direct) omits that factor, treating the tanh-space gradient as if
it were the alpha-space gradient directly. This is imprecise, not merely
a style difference -- SLSQP's local QP step is genuinely misled by it
near the box boundary, where the true factor is large -- though
evidently not incorrect enough to have changed any previously-reported
*result*, since the regression tests here reproduce every prior exact
value bit for bit regardless.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from .forward import a_from_alphas, forward_with_grad, grad_free_vars
from .sdp_design import _padded_alpha_seeds

Interval = tuple[float, float]


def _gamma_to_alphas(gamma_free: np.ndarray) -> np.ndarray:
    """gamma_free = tanh(alpha_free) in (-1,1); alpha_n is solved for so
    that sum(alphas)=0 (constraint D) holds automatically."""
    alphas_free = np.arctanh(gamma_free)
    alpha_n = -np.sum(alphas_free)
    return np.concatenate([alphas_free, [alpha_n]])


def _chain_rule_scale(gamma_free: np.ndarray) -> np.ndarray:
    """d(alpha_free_j)/d(gamma_free_j) = 1/(1-gamma_free_j^2), the arctanh
    derivative -- see module docstring for why this must be applied
    explicitly to every gradient built on gamma_free."""
    return 1.0 / (1.0 - gamma_free ** 2)


# --------------------------------------------------------------------------
# Phase 1: open the gap
# --------------------------------------------------------------------------

@dataclass
class Phase1Result:
    alpha: np.ndarray | None
    kappa_min: float
    converged: bool


def phase1_maximize_depth(n: int, I0: Sequence[Interval], sigma: Sequence[int], mu0: float,
                           theta_grid: Sequence[np.ndarray], gamma_bound: float = 0.9995,
                           maxiter: int = 400, n_steps: int = 20, max_bisections: int = 4,
                           perturb_scales: Sequence[float] = (1e-3, 1e-2, 0.05, 0.1, 0.2),
                           seed: int = 0) -> Phase1Result:
    """One sign pattern's Phase 1 (spec Sec. 3.2), as a RAMPED-TARGET
    feasibility search rather than the literal unconstrained epigraph
    maximisation -- a deliberate, user-approved deviation (Authority rule,
    spec Sec. 0), not a silent one. The literal "max_{alpha,t} t
    s.t. sigma_i*kappa_B>=t" has no interior optimum: more contrast always
    helps kappa_min, so SLSQP drives every free alpha_j to exactly the box
    edge regardless of how loose that box is (confirmed empirically on the
    n=7 multiband case: alpha components pinned at +-arctanh(gamma_bound),
    kappa_min reaching ~2.2e9, and the resulting Phase 2 seed giving
    delta=57.9 instead of the known-good 0.0277671370191698). Bounding the
    reparametrization does not fix this -- it only moves where the
    runaway stops.

    Instead: ramp the REQUIRED depth target from 0 up to mu0 in stages,
    each a feasibility problem (small quadratic regularizer toward
    gamma_free=0, weight tiny relative to the constraint) warm-started
    from the previous stage; this reliably reaches kappa_min~cosh(mu0)
    without an incentive to overshoot it. The full target is tried
    directly first (found empirically more reliable than ramping in small
    steps from a point right next to alpha=0's own degeneracy), ramping
    used only as a fallback, with the step halved (up to max_bisections
    times) whenever a stage fails to verify.

    alpha=0 is itself a genuine gradient degeneracy (Q-1=|q2|^2 has its
    global minimum there, and by the identical L_j K p^(j) structure so
    does d(kappa)/d(alpha_j) -- every A_j is diagonal at alpha=0), so a
    small perturbation is used as the actual starting point; a single
    fixed scale is not always enough for larger targets, so several are
    tried and the one reaching the largest depth kept.
    """
    theta_C = list(theta_grid)
    bounds = [(-gamma_bound, gamma_bound)] * n
    reg = 1e-4
    coshmu0 = np.cosh(mu0)

    def objective(gamma_free):
        return reg * float(np.sum(gamma_free ** 2))

    def objective_grad(gamma_free):
        return 2.0 * reg * gamma_free

    def make_constraints(coshmu_k):
        constraints = []
        for th, sig in zip(theta_C, sigma):
            def fun(gamma_free, th=th, sig=sig):
                alphas = _gamma_to_alphas(gamma_free)
                kap = forward_with_grad(alphas, th)["kappa"]
                return sig * kap - coshmu_k

            def jac(gamma_free, th=th, sig=sig):
                alphas = _gamma_to_alphas(gamma_free)
                res = forward_with_grad(alphas, th)
                dkap_dgamma = grad_free_vars(sig * res["dkappa"]) * _chain_rule_scale(gamma_free)[:, None]
                return dkap_dgamma.T

            constraints.append({"type": "ineq", "fun": fun, "jac": jac})
        return constraints

    def verify(gamma_free, coshmu_k, tol=1e-4):
        alphas = _gamma_to_alphas(gamma_free)
        for th, sig in zip(theta_C, sigma):
            if np.min(sig * forward_with_grad(alphas, th)["kappa"]) < coshmu_k - tol:
                return False
        return True

    def try_stage(gamma_start, mu_k):
        coshmu_k = np.cosh(mu_k)
        res = minimize(objective, gamma_start, jac=objective_grad, constraints=make_constraints(coshmu_k),
                        bounds=bounds, method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-14})
        if not np.all(np.isfinite(res.x)):
            return None
        return res.x if verify(res.x, coshmu_k) else None

    def depth_of(gamma_free):
        alphas = _gamma_to_alphas(gamma_free)
        return min(np.min(sig * forward_with_grad(alphas, th)["kappa"])
                    for th, sig in zip(theta_C, sigma))

    def one_attempt(gamma0):
        direct = try_stage(gamma0, mu0)
        if direct is not None:
            return direct, depth_of(direct)

        gamma_free = gamma0
        mu_prev = 0.0
        schedule = list(np.linspace(0.0, mu0, n_steps + 1)[1:]) if n_steps > 0 else [mu0]
        idx = 0
        while idx < len(schedule):
            mu_target = schedule[idx]
            lo, hi = mu_prev, mu_target
            outcome = try_stage(gamma_free, hi)
            bisections = 0
            while outcome is None and bisections < max_bisections:
                hi = (lo + hi) / 2.0
                outcome = try_stage(gamma_free, hi)
                bisections += 1
            if outcome is None:
                break
            gamma_free, mu_prev = outcome, hi
            if hi >= mu_target - 1e-12:
                idx += 1
        return gamma_free, depth_of(gamma_free)

    rng = np.random.default_rng(seed)
    best = None
    for scale in perturb_scales:
        gamma0 = rng.normal(0.0, scale, n)
        gamma_result, depth = one_attempt(gamma0)
        if best is None or depth > best[1]:
            best = (gamma_result, depth)
        if depth >= coshmu0 - 1e-6:
            break

    gamma_free, kmin = best
    return Phase1Result(alpha=_gamma_to_alphas(gamma_free), kappa_min=kmin, converged=True)


def phase1_all_sign_patterns(n: int, I0: Sequence[Interval], mu0: float,
                              theta_grid: Sequence[np.ndarray], gamma_bound: float = 0.9995,
                              maxiter: int = 400, seed: int = 0) -> dict:
    """Runs phase1_maximize_depth's ramped-target search for EVERY sigma in
    {+-1}^m0, independently -- the direct source of the kappa_min_by_sigma
    emission (spec Sec. 6). winner = the single feasible pattern
    (kappa_min >= cosh(mu0), up to the ramped search's own 1e-4 verify
    tolerance) with the LARGEST kappa_min, per spec Sec. 3.2's literal
    "the pattern with the largest value is passed to Phase 2"; None if no
    pattern reaches the target."""
    coshmu0 = np.cosh(mu0)
    kappa_min_by_sigma: dict[tuple, float] = {}
    alpha_by_sigma: dict[tuple, np.ndarray | None] = {}
    for sigma in itertools.product((1, -1), repeat=len(I0)):
        res = phase1_maximize_depth(n, I0, sigma, mu0, theta_grid, gamma_bound=gamma_bound,
                                     maxiter=maxiter, seed=seed)
        kappa_min_by_sigma[sigma] = res.kappa_min
        alpha_by_sigma[sigma] = res.alpha

    feasible = [s for s, k in kappa_min_by_sigma.items() if k >= coshmu0 - 1e-4]
    winner = max(feasible, key=lambda s: kappa_min_by_sigma[s]) if feasible else None
    return {
        "kappa_min_by_sigma": kappa_min_by_sigma,
        "alpha_by_sigma": alpha_by_sigma,
        "feasible": feasible,
        "winner": winner,
        "winner_alpha": alpha_by_sigma.get(winner) if winner is not None else None,
    }


# --------------------------------------------------------------------------
# Phase 2: flatten the pass band
# --------------------------------------------------------------------------

@dataclass
class Phase2Result:
    alpha: np.ndarray | None
    delta: float | None
    start_origin: str | None
    per_start: dict = field(default_factory=dict)


def build_seed_pool(n: int, phase1_alpha: np.ndarray | None,
                     smaller_solutions: dict[int, np.ndarray] | None = None,
                     magnitude_seed: np.ndarray | None = None,
                     n_random: int = 3, gamma_bound: float = 0.9995,
                     seed: int = 0) -> dict[str, np.ndarray]:
    """The Phase 2 start pool (spec Sec. 3.2): Phase-1 point; zero-padded
    embeddings of designs retained at lower degrees; a small number of
    random draws -- plus, OFF unless magnitude_seed is given, a fourth
    investigative member (the magnitude-SDP factorisation) purely to
    answer the manuscript's own open question about that warm start
    (tracked via start_origin, spec Sec. 6). No member is screened for
    (C)-feasibility here or anywhere downstream."""
    rng = np.random.default_rng(seed)
    pool: dict[str, np.ndarray] = {}

    if phase1_alpha is not None:
        pool["phase1"] = np.clip(np.tanh(phase1_alpha[:n]), -gamma_bound, gamma_bound)

    if smaller_solutions:
        for n_small, alphas_small in smaller_solutions.items():
            if n_small >= n:
                continue
            padded_variants = _padded_alpha_seeds(np.asarray(alphas_small, dtype=float), n, rng)
            for i, padded in enumerate(padded_variants):
                pool[f"zero_pad_from_n{n_small}_{i}"] = np.clip(np.tanh(padded[:n]), -gamma_bound, gamma_bound)

    if magnitude_seed is not None:
        pool["magnitude_sdp_seed"] = np.clip(np.tanh(np.asarray(magnitude_seed, dtype=float)[:n]),
                                              -gamma_bound, gamma_bound)

    for i in range(n_random):
        pool[f"random_{i}"] = np.clip(0.02 * rng.standard_normal(n), -gamma_bound, gamma_bound)

    if not pool:
        pool["zero"] = np.zeros(n)
    return pool


def phase2_flatten(n: int, I0: Sequence[Interval], I1: Sequence[Interval], mu0: float,
                    sigma: Sequence[int], seed_pool: dict[str, np.ndarray],
                    theta_grid_B: Sequence[np.ndarray], theta_grid_C: Sequence[np.ndarray],
                    gamma_bound: float = 0.9995, maxiter: int = 400) -> Phase2Result:
    """From the seed pool, solve the grid form of spec eq:direct by SQP
    with exact gradients:

        min_{alpha,t} t   s.t.  Q(theta)-1 <= t          (theta in I1 grid)
                              sigma_i*kappa(theta) >= cosh(mu0)  (theta in I0 grid)

    t IS delta directly (manuscript eq. 6.6's own convention, no
    sqrt(u)-1 bookkeeping). Runs every named seed (no C-feasibility
    screening); keeps the best (minimum delta) among converged, verified
    results."""
    theta_B = list(theta_grid_B)
    theta_C = list(theta_grid_C)
    coshmu0 = np.cosh(mu0)

    def unpack(x):
        return x[:-1], x[-1]

    def objective(x):
        return x[-1]

    def objective_grad(x):
        g = np.zeros(len(x))
        g[-1] = 1.0
        return g

    constraints = []
    for th in theta_B:
        def fun(x, th=th):
            gamma_free, t = unpack(x)
            alphas = _gamma_to_alphas(gamma_free)
            Q = forward_with_grad(alphas, th)["Q"]
            return t - (Q - 1.0)

        def jac(x, th=th):
            gamma_free, t = unpack(x)
            alphas = _gamma_to_alphas(gamma_free)
            res = forward_with_grad(alphas, th)
            dQ_dgamma = grad_free_vars(res["dQ"]) * _chain_rule_scale(gamma_free)[:, None]
            J = np.zeros((len(th), len(x)))
            J[:, :-1] = -dQ_dgamma.T
            J[:, -1] = 1.0
            return J

        constraints.append({"type": "ineq", "fun": fun, "jac": jac})

    for th, sig in zip(theta_C, sigma):
        def fun(x, th=th, sig=sig):
            gamma_free, t = unpack(x)
            alphas = _gamma_to_alphas(gamma_free)
            kap = forward_with_grad(alphas, th)["kappa"]
            return sig * kap - coshmu0

        def jac(x, th=th, sig=sig):
            gamma_free, t = unpack(x)
            alphas = _gamma_to_alphas(gamma_free)
            res = forward_with_grad(alphas, th)
            dkap_dgamma = grad_free_vars(sig * res["dkappa"]) * _chain_rule_scale(gamma_free)[:, None]
            J = np.zeros((len(th), len(x)))
            J[:, :-1] = dkap_dgamma.T
            return J

        constraints.append({"type": "ineq", "fun": fun, "jac": jac})

    bounds = [(-gamma_bound, gamma_bound)] * n + [(0.0, None)]

    def verify(gamma_free, t, tol=1e-6):
        # A genuinely useful filter has delta bounded by a modest constant
        # (never remotely close to this) -- SLSQP occasionally converges a
        # badly-scaled seed to a "locally stationary" point that technically
        # satisfies the pointwise inequalities on the grid with an enormous
        # t; reject rather than report it as a successful design.
        if t > 100.0:
            return False
        alphas = _gamma_to_alphas(gamma_free)
        for th in theta_B:
            Q = forward_with_grad(alphas, th)["Q"]
            if np.max(Q - 1.0) > t + tol:
                return False
        for th, sig in zip(theta_C, sigma):
            kap = forward_with_grad(alphas, th)["kappa"]
            if np.min(sig * kap) < coshmu0 - tol:
                return False
        return True

    best = None
    per_start: dict[str, float | None] = {}
    for name, gamma0 in seed_pool.items():
        gamma0 = np.clip(np.asarray(gamma0, dtype=float), -gamma_bound, gamma_bound)
        alphas0 = _gamma_to_alphas(gamma0)
        Qmax0 = max((np.max(forward_with_grad(alphas0, th)["Q"]) for th in theta_B), default=1.0)
        t0 = max(0.0, Qmax0 - 1.0)
        x0 = np.concatenate([gamma0, [t0]])
        res = minimize(objective, x0, jac=objective_grad, constraints=constraints, bounds=bounds,
                        method="SLSQP", options={"maxiter": maxiter, "ftol": 1e-14})
        if not np.all(np.isfinite(res.x)):
            per_start[name] = None
            continue
        gamma_try, t_try = unpack(res.x)
        if verify(gamma_try, t_try):
            per_start[name] = float(t_try)
            if best is None or t_try < best[1]:
                best = (gamma_try, float(t_try), name)
        else:
            per_start[name] = None

    if best is None:
        return Phase2Result(alpha=None, delta=None, start_origin=None, per_start=per_start)
    gamma_free, delta, name = best
    return Phase2Result(alpha=_gamma_to_alphas(gamma_free), delta=delta, start_origin=name, per_start=per_start)


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

@dataclass
class DirectDesignResult:
    n: int
    mu0: float
    status: str                            # "optimal" / "phase1_infeasible" / "phase2_infeasible"
    sigma_star: tuple | None = None
    alpha: np.ndarray | None = None
    a: np.ndarray | None = None
    impedances: np.ndarray | None = None
    delta: float | None = None
    kappa_min_by_sigma: dict = field(default_factory=dict)
    start_origin: str | None = None
    per_start: dict = field(default_factory=dict)


def design_direct_literal(n: int, I0: Sequence[Interval], I1: Sequence[Interval], mu0: float,
                           smaller_solutions: dict[int, np.ndarray] | None = None,
                           magnitude_seed: np.ndarray | None = None,
                           n_grid_B: int = 200, n_grid_C: int = 200,
                           gamma_bound: float = 0.9995, maxiter: int = 400,
                           seed: int = 0) -> DirectDesignResult:
    """Stage 2's top-level orchestrator, one degree n: Phase 1 (all sign
    patterns) -> Phase 2 for EVERY Phase-1-feasible pattern, keeping the
    best (min delta) -- a user-approved, flagged deviation from spec Sec.
    3.2's literal "the pattern with the largest value is passed to Phase
    2" (see direct.py's module docstring): confirmed on the n=7 multiband
    case that the largest-margin pattern is not always the one that
    actually flattens well (sigma=(-1,-1), kappa_min=1.926, gave
    delta=57.9, while the discarded sigma=(-1,1), kappa_min=1.00125, gave
    the correct delta=0.0277671...). Cost is modest in practice: m_0 is
    typically 1-3 per the manuscript's own instances, so at most 2^m_0<=8
    Phase 2 solves, not a drastic overhead. This is what the eventual
    driver calls once per degree; see the module docstring for how this
    (and the rest of this function) differs, deliberately, from the
    earlier sdp_design.design_direct."""
    theta_B = [np.linspace(lo, hi, n_grid_B) for lo, hi in I1]
    theta_C = [np.linspace(lo, hi, n_grid_C) for lo, hi in I0]

    if I0:
        p1 = phase1_all_sign_patterns(n, I0, mu0, theta_C, gamma_bound=gamma_bound,
                                       maxiter=maxiter, seed=seed)
        kappa_min_by_sigma = p1["kappa_min_by_sigma"]
        if not p1["feasible"]:
            return DirectDesignResult(n=n, mu0=mu0, status="phase1_infeasible",
                                       kappa_min_by_sigma=kappa_min_by_sigma)
        candidate_sigmas = p1["feasible"]
        phase1_alpha_by_sigma = p1["alpha_by_sigma"]
    else:
        candidate_sigmas = [()]
        phase1_alpha_by_sigma = {(): np.zeros(n + 1)}
        kappa_min_by_sigma = {}

    best: tuple[float, Phase2Result, tuple] | None = None
    per_start_by_sigma: dict[tuple, dict] = {}
    for sigma in candidate_sigmas:
        seed_pool = build_seed_pool(n, phase1_alpha_by_sigma[sigma], smaller_solutions, magnitude_seed,
                                     gamma_bound=gamma_bound, seed=seed)
        p2 = phase2_flatten(n, I0, I1, mu0, sigma, seed_pool, theta_B, theta_C,
                             gamma_bound=gamma_bound, maxiter=maxiter)
        per_start_by_sigma[sigma] = p2.per_start
        if p2.alpha is not None and (best is None or p2.delta < best[0]):
            best = (p2.delta, p2, sigma)

    if best is None:
        return DirectDesignResult(n=n, mu0=mu0, status="phase2_infeasible",
                                   kappa_min_by_sigma=kappa_min_by_sigma,
                                   per_start={str(s): ps for s, ps in per_start_by_sigma.items()})

    _, p2, sigma_star = best
    a = a_from_alphas(p2.alpha)
    impedances = np.empty(n + 2)
    impedances[0] = 1.0
    for j, alpha_j in enumerate(p2.alpha):
        impedances[j + 1] = impedances[j] * np.exp(alpha_j)

    return DirectDesignResult(n=n, mu0=mu0, status="optimal", sigma_star=sigma_star,
                               alpha=p2.alpha, a=a, impedances=impedances, delta=p2.delta,
                               kappa_min_by_sigma=kappa_min_by_sigma,
                               start_origin=p2.start_origin,
                               per_start={str(s): ps for s, ps in per_start_by_sigma.items()})
