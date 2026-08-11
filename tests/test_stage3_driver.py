"""Stage 3 driver (scattering/driver.py, spec Sec. 6 / Sec. 9 deliverable
2): validation (rule 7), the gamma/n_min_green closed-form (Sec. 5.4),
the rule-4 retention logic, CSV/LaTeX output, and one live (slow) degree
run end to end through direct.py + certify.py + dual_certify.py together.

Run: python3 -m pytest tests/test_stage3_driver.py -v   (from repo root)
"""
import csv
import json
import os

import numpy as np
import pytest

from scattering.driver import (
    DegreeRecord, validate_intervals, gamma_geometric, n_min_green,
    retained_degree_record, save_records_csv, save_records_latex, run_degree_stage3,
)


# --------------------------------------------------------------------------
# rule 7: input validation
# --------------------------------------------------------------------------

def test_validate_intervals_accepts_a_valid_configuration():
    validate_intervals([(np.pi / 6, np.pi / 4)], [(np.pi / 2, 2.5)])  # must not raise


@pytest.mark.parametrize("I0,I1", [
    ([(-0.1, 0.5)], [(1.0, 2.0)]),          # out of [0,pi]
    ([(0.0, 0.5)], [(1.0, 2.0)]),           # 0 in I0
    ([(0.5, 1.0)], [(2.0, np.pi)]),         # pi in I1
    ([(0.5, 1.5)], [(1.0, 2.0)]),           # overlapping
    ([(0.5, 1.0)], [(1.0, 2.0)]),           # touching, zero gap
])
def test_validate_intervals_rejects_rule7_violations(I0, I1):
    with pytest.raises(ValueError):
        validate_intervals(I0, I1)


# --------------------------------------------------------------------------
# Sec. 5.4: gamma / n_min_green
# --------------------------------------------------------------------------

def test_gamma_geometric_matches_hand_computation():
    I1 = [(np.pi / 2, 2.5)]
    I0 = [(np.pi / 6, np.pi / 4)]
    a_E, b_E = np.cos(2.5), np.cos(np.pi / 2)
    x_lo, x_hi = np.cos(np.pi / 4), np.cos(np.pi / 6)
    mid = (a_E + b_E) / 2.0
    candidates = [x_lo, x_hi] + ([mid] if x_lo < mid < x_hi else [])
    expected_ratio = min(abs((2 * x - a_E - b_E) / (b_E - a_E)) for x in candidates)
    expected = np.arccosh(expected_ratio) if expected_ratio >= 1.0 else None

    g = gamma_geometric(I0, I1)
    if expected is None:
        assert g is None
    else:
        assert g is not None
        assert abs(g - expected) < 1e-12


def test_gamma_geometric_none_for_multi_interval_I1():
    I1 = [(1.0, 1.5), (2.0, 2.5)]
    I0 = [(0.2, 0.4)]
    assert gamma_geometric(I0, I1) is None


def test_gamma_geometric_none_for_empty_I0():
    assert gamma_geometric([], [(1.0, 1.5)]) is None


def test_n_min_green_formula():
    gamma, mu0, delta = 1.5, 0.05, 0.01
    expected = np.log(np.sinh(mu0) ** 2 / delta) / gamma
    assert abs(n_min_green(gamma, mu0, delta) - expected) < 1e-12


@pytest.mark.parametrize("gamma,mu0,delta", [
    (None, 0.05, 0.01), (0.0, 0.05, 0.01), (1.5, 0.05, None), (1.5, 0.05, 0.0), (1.5, 0.05, -0.01),
])
def test_n_min_green_none_on_degenerate_input(gamma, mu0, delta):
    assert n_min_green(gamma, mu0, delta) is None


# --------------------------------------------------------------------------
# rule 4: retention = least admissible degree, not least delta
# --------------------------------------------------------------------------

def _fake_record(n, admissible, delta_achieved):
    return DegreeRecord(
        n=n, status="optimal", underline_delta=1e-6, dual_ray_found=False,
        dual_status="dual_certified", dual_blocks_all_feasible=True,
        delta_achieved=delta_achieved, kappa_min=1.5, mu_min=0.9, s_0=0.5,
        sigma_star=(-1,), kappa_min_by_sigma={(-1,): 1.5, (1,): 0.2}, admissible=admissible,
        N_required=10.0, alpha=[0.1, -0.1], rho=[1.0, 1.1, 1.0], start_origin="phase1",
        grid_vs_exact={"delta_discrepancy": 1e-8, "kappa_min_discrepancy": 1e-9},
        gamma=1.2, n_min_green=4.5, time_direct_s=0.1, time_sdp_s=0.1,
    )


def test_retained_degree_record_picks_least_admissible_n_not_least_delta():
    records = [
        _fake_record(3, admissible=False, delta_achieved=10.0),
        _fake_record(4, admissible=True, delta_achieved=0.05),   # least admissible n
        _fake_record(5, admissible=True, delta_achieved=0.001),  # smallest delta, but NOT retained
    ]
    ret = retained_degree_record(records)
    assert ret is not None
    assert ret.n == 4


def test_retained_degree_record_none_when_nothing_admissible():
    records = [_fake_record(3, admissible=False, delta_achieved=1.0),
               _fake_record(4, admissible=False, delta_achieved=0.5)]
    assert retained_degree_record(records) is None


# --------------------------------------------------------------------------
# CSV / LaTeX output (deliverable 2)
# --------------------------------------------------------------------------

def test_save_records_csv_and_latex_roundtrip(tmp_path):
    records = [_fake_record(3, admissible=False, delta_achieved=10.0),
               _fake_record(4, admissible=True, delta_achieved=0.05)]
    csv_path = str(tmp_path / "scan.csv")
    tex_path = str(tmp_path / "scan.tex")
    save_records_csv(records, csv_path)
    save_records_latex(records, tex_path)

    assert os.path.exists(csv_path) and os.path.exists(tex_path)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["n"] == "3"
    assert json.loads(rows[0]["alpha"]) == [0.1, -0.1]

    with open(tex_path) as f:
        tex_lines = f.readlines()
    assert len(tex_lines) == 2
    assert tex_lines[1].strip().endswith(r"\\")


# --------------------------------------------------------------------------
# One live (slow) integration run: direct.py + certify.py + dual_certify.py
# wired together through run_degree_stage3, on a case fast enough to reach
# phase1_infeasible (mu0 far beyond what n=1 can open) rather than running
# a full multi-start Phase 2 search.
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_run_degree_stage3_end_to_end_infeasible_case():
    I0 = [(np.pi / 6, np.pi / 4)]
    I1 = [(np.pi / 2, 2.5)]
    rec = run_degree_stage3(1, I0, I1, 50.0, 1e-2, 1e-2, n_grid_B=30, n_grid_C=30)

    assert rec.status == "phase1_infeasible"
    assert rec.delta_achieved is None
    assert rec.admissible is False
    # the dual side is a SEPARATE solve (dual_certify has no notion of
    # direct.py's own Phase 1/2 outcome) and should still report cleanly
    assert rec.dual_status in ("dual_certified", "dual_ray_found", "dual_infeasible_to_certify")
    assert rec.gamma is not None and rec.gamma > 0.0
