"""Experiment driver (spec Sec. 8): scan n, emit one JSON-serializable
record per degree with the exact schema given in the spec, plus the two
paper experiments (spec Sec. 8, "Two experiments for the paper").

VALIDATION POLICY (explicit user correction, not a stylistic choice): the
records produced here must NOT be checked against the manuscript's own
numeric worked-example tables -- "do not check against the table values;
they are not certain!" -- since those tables were computed under the old,
now-purged sqrt(u)-1 objective convention and are not trustworthy ground
truth. Validation is via internal consistency instead:
  - delta_mag(n) should be monotone non-increasing in n (lower_bounds'
    own consistency check, spec Sec. 7): a bigger block can only relax
    the magnitude-SDP feasible set, never shrink it.
  - delta_mag(n) <= delta_achieved(n) always (design_sdp_magnitude is a
    genuine relaxation of design_direct's feasible set).
  - delta_certified should match delta_achieved from design_direct to
    within the grid/solver tolerance both were computed under (they read
    off the same trig polynomial, just one exactly and one on a grid).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from .sdp_design import (
    design_sdp_magnitude, design_direct, design_filter_full,
    _factorize_magnitude_f, _seed_gap_quality, MagnitudeSDPResult, DirectDesignResult,
)
from .certify import certify

Interval = tuple[float, float]


def _jsonable(x):
    """Recursively convert numpy scalars/arrays (and None) into plain
    JSON-serializable Python types."""
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


@dataclass
class DegreeRecord:
    """Exactly the fields of the spec's own JSON schema (Sec. 8)."""
    n: int
    mu_0: float
    delta_mag: float | None
    delta_lift: float | None
    delta_achieved: float | None
    winning_start: str | None
    per_start: dict | None
    sdp_seed_gap_depth: float | None
    cosh_mu_0: float
    alpha: list | None
    impedances: list | None
    sum_alpha_residual: float | None
    mu_min_certified: float | None
    delta_certified: float | None
    time_sdp_s: float
    time_phase1_s: float | None
    time_phase2_s: float | None
    sigma: list | None
    status: str  # not in the spec's example, but needed to tell infeasible/failed degrees apart

    def to_json_dict(self) -> dict:
        return _jsonable(asdict(self))


def run_degree(n: int, J0: Sequence[Interval], J1: Sequence[Interval], mu0: float,
                smaller_solutions: dict[int, np.ndarray] | None = None,
                include_lift: bool = False, seed: int = 0,
                N_values: Sequence[int] = ()) -> DegreeRecord:
    """One degree's worth of the driver: magnitude SDP (bound + optional
    warm-start seed) -> design_direct (the achieved design) -> certify
    (exact numbers, independent of whatever grid design_direct used).
    Optionally also the lifted SDP, kept behind include_lift (spec Sec. 7,
    "keep the lifted SDP behind a flag").
    """
    t0 = time.time()
    mag_res = design_sdp_magnitude(n, J0, J1, mu0)
    time_sdp_s = time.time() - t0

    delta_mag = mag_res.delta_mag if mag_res.status == "optimal" else None

    sdp_seed_gap_depth = None
    if mag_res.status == "optimal" and mag_res.f is not None:
        try:
            _, a_sdp, info = _factorize_magnitude_f(mag_res.f)
            if info.reliable:
                sdp_seed_gap_depth = _seed_gap_quality(a_sdp, J0)
        except Exception:
            pass

    delta_lift = None
    if include_lift:
        lift_res = design_filter_full(n, J0, J1, mu0)
        if lift_res is not None and lift_res.status in ("optimal", "optimal_inaccurate") and lift_res.delta is not None:
            delta_lift = lift_res.delta

    direct_res = design_direct(
        n, J0, J1, mu0,
        smaller_solutions=smaller_solutions,
        magnitude_sdp_result=mag_res if mag_res.status == "optimal" else None,
        seed=seed,
    )

    delta_achieved = direct_res.delta if direct_res.status == "optimal" else None
    alpha = direct_res.alphas if direct_res.status == "optimal" else None
    impedances = direct_res.impedances if direct_res.status == "optimal" else None
    sigma = list(direct_res.sigma) if direct_res.sigma is not None else None
    sum_alpha_residual = float(np.sum(alpha)) if alpha is not None else None

    mu_min_certified = None
    delta_certified = None
    if direct_res.status == "optimal" and alpha is not None:
        cert = certify(alpha, J0, J1, sigma, N_values=N_values)
        mu_min_certified = cert.mu_min
        delta_certified = cert.delta

    return DegreeRecord(
        n=n, mu_0=mu0,
        delta_mag=delta_mag, delta_lift=delta_lift, delta_achieved=delta_achieved,
        winning_start=direct_res.winning_start, per_start=direct_res.per_start,
        sdp_seed_gap_depth=sdp_seed_gap_depth, cosh_mu_0=float(np.cosh(mu0)),
        alpha=alpha, impedances=impedances, sum_alpha_residual=sum_alpha_residual,
        mu_min_certified=mu_min_certified, delta_certified=delta_certified,
        time_sdp_s=time_sdp_s, time_phase1_s=direct_res.time_phase1_s, time_phase2_s=direct_res.time_phase2_s,
        sigma=sigma, status=direct_res.status,
    )


def degree_scan_direct(n_range: Sequence[int], J0: Sequence[Interval], J1: Sequence[Interval], mu0: float,
                        include_lift: bool = False, seed: int = 0,
                        N_values: Sequence[int] = ()) -> list[DegreeRecord]:
    """Scan the full predefined n_range (NOT stop-at-first-feasible, spec
    Sec. 8), keeping the best result via degree_scan's own selection logic
    downstream -- every record is kept, not just the best, so the driver's
    output documents the whole scan. Each degree's design_direct call is
    warm-started from the best *achieved* design at any smaller degree
    already scanned (spec's smaller_solutions mechanism)."""
    records: list[DegreeRecord] = []
    smaller_solutions: dict[int, np.ndarray] = {}
    for n in n_range:
        rec = run_degree(n, J0, J1, mu0, smaller_solutions=smaller_solutions,
                          include_lift=include_lift, seed=seed, N_values=N_values)
        records.append(rec)
        if rec.status == "optimal" and rec.alpha is not None:
            smaller_solutions[n] = np.asarray(rec.alpha, dtype=float)
    return records


def best_degree_record(records: Sequence[DegreeRecord]) -> DegreeRecord | None:
    """Smallest delta_achieved among degrees that actually reached
    "optimal" (spec Sec. 8: scan the whole range, keep the best result,
    not the first feasible one)."""
    feasible = [r for r in records if r.status == "optimal" and r.delta_achieved is not None]
    if not feasible:
        return None
    return min(feasible, key=lambda r: r.delta_achieved)


def check_delta_mag_monotone(records: Sequence[DegreeRecord], tol: float = 1e-6) -> bool:
    """Internal-consistency check (spec Sec. 7): delta_mag(n) must be
    monotone non-increasing in n, and delta_mag(n) <= delta_achieved(n)
    whenever both are available -- NOT a check against manuscript table
    values (see module docstring)."""
    ok = True
    prev = None
    for rec in sorted(records, key=lambda r: r.n):
        if rec.delta_mag is not None:
            if prev is not None and rec.delta_mag > prev + tol:
                ok = False
            prev = rec.delta_mag
        if rec.delta_mag is not None and rec.delta_achieved is not None:
            if rec.delta_mag > rec.delta_achieved + tol:
                ok = False
    return ok


def save_records(records: Sequence[DegreeRecord], path: str) -> None:
    with open(path, "w") as f:
        json.dump([r.to_json_dict() for r in records], f, indent=2)
