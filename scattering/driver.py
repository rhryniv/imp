"""Stage 3 degree-scan driver (spec Sec. 6 "Required emissions" / Sec. 9
deliverable 2): wires direct.py (Stage 2's two-phase optimiser),
certify.py (exact certification) and dual_certify.py (Stage 3's explicit
dual-feasible-point / dual-improving-ray certificates) into one record
per degree, matching the spec's own emissions table field-for-field,
scanned in increasing n (rule 5: L=2n is tied to the degree, and the
Phase-2 start pool at degree n depends on every smaller degree already
scanned).

Supersedes experiment.py's run_degree/degree_scan_direct/DegreeRecord,
which call the pre-Stage-2/3 legacy sdp_design.py pipeline (design_direct,
design_sdp_magnitude's raw, UNCERTIFIED primal value reported directly as
if it were a bound -- exactly what rule 2 forbids). experiment.py is left
in place only because Stage 1's own two paper experiments were already
run and reported against it; new work should use this module.

IMPORTANT, rule 2 in this driver's own terms: `delta_mag` from the old
driver is gone. The bound emitted here is `underline_delta`, from an
explicit dual-feasible point (dual_certify.solve_magnitude_sdp_with_duals),
never the SDP solver's raw primal objective.

IMPORTANT, rule 4, in this driver's own terms: the RETAINED degree is the
LEAST n with `admissible=True`, not the n with the smallest
`delta_achieved` -- `retained_degree_record` implements exactly this, and
callers should not substitute `min(records, key=delta_achieved)`.
"""
from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from .direct import design_direct_literal
from .certify import certify_exact, grid_vs_exact
from .dual_certify import solve_magnitude_sdp_with_duals

Interval = tuple[float, float]


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def validate_intervals(I0: Sequence[Interval], I1: Sequence[Interval]) -> None:
    """Rule 7: I0, I1 subset of [0,pi], disjoint with a positive gap,
    0 not in I0, pi not in I1."""
    all_intervals = [("I0", iv) for iv in I0] + [("I1", iv) for iv in I1]
    for name, (lo, hi) in all_intervals:
        if not (0.0 <= lo < hi <= np.pi):
            raise ValueError(f"{name} interval ({lo},{hi}) not a valid subinterval of [0,pi]")
    for lo, hi in I0:
        if lo <= 0.0 <= hi:
            raise ValueError(f"0 in I0 interval ({lo},{hi}) -- forbidden by rule 7")
    for lo, hi in I1:
        if lo <= np.pi <= hi:
            raise ValueError(f"pi in I1 interval ({lo},{hi}) -- forbidden by rule 7")
    ordered = sorted(iv for _, iv in all_intervals)
    for (lo1, hi1), (lo2, hi2) in zip(ordered, ordered[1:]):
        if hi1 >= lo2:
            raise ValueError(f"I0/I1 intervals ({lo1},{hi1}) and ({lo2},{hi2}) are not disjoint "
                              "with a positive gap -- forbidden by rule 7")


def gamma_geometric(I0: Sequence[Interval], I1: Sequence[Interval]) -> float | None:
    """Spec Sec. 5.4: gamma = arccosh(min_{x in F} |(2x-a-b)/(b-a)|), for
    E=[a,b] the image of the single I1 component under x=cos(theta) and F
    the image of I0. Requires m1=1 (single-interval I1); returns None
    otherwise, or if the geometry doesn't yield an arccosh-able ratio
    (>=1) -- an ill-posed instance for this particular formula, not a
    bug to paper over."""
    if len(I1) != 1 or not I0:
        return None
    theta_a, theta_b = I1[0]
    a_E, b_E = float(np.cos(theta_b)), float(np.cos(theta_a))  # cos decreasing => a_E < b_E
    mid = (a_E + b_E) / 2.0

    def g(x):
        return abs((2.0 * x - a_E - b_E) / (b_E - a_E))

    candidates = []
    for lo, hi in I0:
        x_lo, x_hi = float(np.cos(hi)), float(np.cos(lo))
        candidates.extend([x_lo, x_hi])
        if x_lo < mid < x_hi:
            candidates.append(mid)
    min_ratio = min(g(x) for x in candidates)
    if min_ratio < 1.0:
        return None
    return float(np.arccosh(min_ratio))


def n_min_green(gamma: float | None, mu0: float, delta: float | None) -> float | None:
    """Spec Sec. 5.4: n_min_green = log(sinh^2(mu_0)/delta) / gamma."""
    if gamma is None or gamma <= 0.0 or delta is None or delta <= 0.0:
        return None
    ratio = np.sinh(mu0) ** 2 / delta
    if ratio <= 0.0:
        return None
    return float(np.log(ratio) / gamma)


@dataclass
class DegreeRecord:
    """Field-for-field the spec's own Sec. 6 emissions table, plus `n`,
    `status`, and per-block dual diagnostics needed to trust
    `underline_delta`/`dual_ray_found` rather than just print them."""
    n: int
    status: str  # "optimal" / "phase1_infeasible" / "phase2_infeasible"
    underline_delta: float | None
    dual_ray_found: bool
    dual_status: str
    dual_blocks_all_feasible: bool
    delta_achieved: float | None          # certify_exact's own exact delta -- NOT the raw SQP/grid value
    kappa_min: float | None
    mu_min: float | None
    s_0: float | None
    sigma_star: tuple | None
    kappa_min_by_sigma: dict
    admissible: bool
    N_required: float | None
    alpha: list | None
    rho: list | None                       # impedances
    start_origin: str | None
    grid_vs_exact: dict
    gamma: float | None
    n_min_green: float | None
    time_direct_s: float
    time_sdp_s: float

    def to_json_dict(self) -> dict:
        return _jsonable(asdict(self))


def run_degree_stage3(n: int, I0: Sequence[Interval], I1: Sequence[Interval], mu0: float,
                       eps0: float, eps1: float,
                       smaller_solutions: dict[int, np.ndarray] | None = None,
                       N_values: Sequence[int] = (), seed: int = 0,
                       n_grid_B: int = 200, n_grid_C: int = 200,
                       admissible_tol: float = 1e-9) -> DegreeRecord:
    """One degree's worth of the Stage 3 driver: direct.py's two-phase
    design -> certify.py's exact certification -> dual_certify.py's
    explicit dual certificate for the magnitude SDP bound. Every field is
    reported even on failure (None/False/empty as appropriate), so a
    failed degree is still a row in the emitted table, not a silent gap.
    """
    t0 = time.time()
    direct_res = design_direct_literal(n, I0, I1, mu0, smaller_solutions=smaller_solutions,
                                        n_grid_B=n_grid_B, n_grid_C=n_grid_C, seed=seed)
    time_direct_s = time.time() - t0

    t0 = time.time()
    dual_res = solve_magnitude_sdp_with_duals(n, I0, I1, mu0)
    time_sdp_s = time.time() - t0
    dual_blocks_all_feasible = all(bc.feasible for bc in dual_res.block_checks) if dual_res.block_checks else False

    alpha = direct_res.alpha
    sigma_star = direct_res.sigma_star
    rho = direct_res.impedances

    cert = None
    grid_vs_exact_diag = {"delta_discrepancy": None, "kappa_min_discrepancy": None}
    if direct_res.status == "optimal" and alpha is not None:
        cert = certify_exact(np.asarray(alpha, dtype=float), I0, I1, sigma_star,
                              N_values=N_values, eps0=eps0, eps1=eps1)
        grid_kappa_min = (direct_res.kappa_min_by_sigma.get(sigma_star)
                           if sigma_star is not None else None)
        grid_vs_exact_diag = grid_vs_exact(direct_res.delta, grid_kappa_min, cert)

    delta_achieved = cert.delta if cert is not None else None
    kappa_min = cert.kappa_min if cert is not None else None
    mu_min = cert.mu_min if cert is not None else None
    s_0 = cert.s_0 if cert is not None else None

    N_required = None
    if mu_min is not None and mu_min > 0.0:
        N_required = float(np.arccosh(eps0 ** -0.5) / mu_min)

    admissible = False
    if kappa_min is not None and delta_achieved is not None and s_0 is not None and s_0 > 0.0:
        cosh_mu0 = float(np.cosh(mu0))
        admissible = (kappa_min >= cosh_mu0 - admissible_tol
                      and delta_achieved <= s_0 * eps1 / (1.0 - eps1) + admissible_tol)

    gamma = gamma_geometric(I0, I1)
    n_min_green_val = n_min_green(gamma, mu0, delta_achieved)

    return DegreeRecord(
        n=n, status=direct_res.status,
        underline_delta=dual_res.underline_delta, dual_ray_found=dual_res.dual_ray_found,
        dual_status=dual_res.status, dual_blocks_all_feasible=dual_blocks_all_feasible,
        delta_achieved=delta_achieved, kappa_min=kappa_min, mu_min=mu_min, s_0=s_0,
        sigma_star=sigma_star, kappa_min_by_sigma=direct_res.kappa_min_by_sigma,
        admissible=admissible, N_required=N_required,
        alpha=alpha, rho=rho, start_origin=direct_res.start_origin,
        grid_vs_exact=grid_vs_exact_diag, gamma=gamma, n_min_green=n_min_green_val,
        time_direct_s=time_direct_s, time_sdp_s=time_sdp_s,
    )


def degree_scan_stage3(n_range: Sequence[int], I0: Sequence[Interval], I1: Sequence[Interval], mu0: float,
                        eps0: float, eps1: float, N_values: Sequence[int] = (),
                        seed: int = 0) -> list[DegreeRecord]:
    """Rule 5: scans n in INCREASING order (n_range is used as given --
    callers must pass it already sorted increasing), each degree's
    direct-design warm-started from every smaller degree's own retained
    alpha already in the scan."""
    validate_intervals(I0, I1)
    records: list[DegreeRecord] = []
    smaller_solutions: dict[int, np.ndarray] = {}
    for n in n_range:
        rec = run_degree_stage3(n, I0, I1, mu0, eps0, eps1,
                                 smaller_solutions=smaller_solutions, N_values=N_values, seed=seed)
        records.append(rec)
        if rec.status == "optimal" and rec.alpha is not None:
            smaller_solutions[n] = np.asarray(rec.alpha, dtype=float)
    return records


def retained_degree_record(records: Sequence[DegreeRecord]) -> DegreeRecord | None:
    """Rule 4: retention is the LEAST admissible degree, not the least
    delta_achieved -- records are scanned in increasing n (as produced by
    degree_scan_stage3) and the first admissible one is returned."""
    for rec in sorted(records, key=lambda r: r.n):
        if rec.admissible:
            return rec
    return None


def save_records_csv(records: Sequence[DegreeRecord], path: str) -> None:
    """Deliverable 2 (spec Sec. 9): a CSV of the Sec. 6 emissions across
    the degree scan. Vector/dict-valued fields (alpha, rho,
    kappa_min_by_sigma, grid_vs_exact) are JSON-encoded into their own
    cell rather than dropped, so the CSV is a complete record."""
    import json
    scalar_fields = ["n", "status", "underline_delta", "dual_ray_found", "dual_status",
                      "dual_blocks_all_feasible", "delta_achieved", "kappa_min", "mu_min", "s_0",
                      "sigma_star", "admissible", "N_required", "start_origin", "gamma", "n_min_green",
                      "time_direct_s", "time_sdp_s"]
    vector_fields = ["alpha", "rho", "kappa_min_by_sigma", "grid_vs_exact"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(scalar_fields + vector_fields)
        for rec in records:
            d = rec.to_json_dict()
            row = [d[k] for k in scalar_fields] + [json.dumps(d[k]) for k in vector_fields]
            w.writerow(row)


def save_records_latex(records: Sequence[DegreeRecord], path: str) -> None:
    """Deliverable 2 (spec Sec. 9): the LaTeX table body (rows only, no
    surrounding tabular/table environment, so the manuscript can own the
    column spec/caption/label) for the scalar-valued emissions."""
    def fmt(x, sig=6):
        if x is None:
            return "--"
        if isinstance(x, bool):
            return r"\checkmark" if x else "--"
        if isinstance(x, float):
            return f"{x:.{sig}g}"
        return str(x)

    lines = []
    for rec in sorted(records, key=lambda r: r.n):
        sigma_str = "".join("+" if s > 0 else "-" for s in rec.sigma_star) if rec.sigma_star else "--"
        cells = [
            str(rec.n), rec.status.replace("_", r"\_"),
            fmt(rec.underline_delta), fmt(rec.delta_achieved),
            fmt(rec.kappa_min), fmt(rec.mu_min), fmt(rec.s_0),
            sigma_str, fmt(rec.admissible), fmt(rec.N_required, sig=4),
            fmt(rec.gamma), fmt(rec.n_min_green, sig=4),
        ]
        lines.append("    " + " & ".join(cells) + r" \\")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
