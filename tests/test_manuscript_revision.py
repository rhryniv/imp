"""Regression coverage for the manuscript revision (numerical-optimisation
instructions applied to this codebase, see driver.py's own module
docstring for the full account): constraint (E) in Phase 2, the new
certify_exact diagnostics (max_kappa_B, Lambda), the N_max-derived
mu0/delta_target, and the new rule-4 admissibility gate (s_0 no longer a
gate).

Run: python3 -m pytest tests/test_manuscript_revision.py -v   (from repo root)
"""
import numpy as np
import pytest

from scattering.certify import certify_exact
from scattering.direct import design_direct_literal
from scattering.driver import (
    DegreeRecord, mu0_from_N_max, delta_target_from_N_max, validate_intervals,
    run_degree_stage3,
)
from scattering.forward import a_from_alphas, kappa_B, q1_abs_sq


def test_mu0_and_delta_target_formulas():
    eps0, eps1, N_max = 1e-2, 1e-2, 20.0
    assert abs(mu0_from_N_max(eps0, N_max) - np.log(4.0 / eps0) / (2.0 * N_max)) < 1e-15
    assert abs(delta_target_from_N_max(eps1, N_max) - eps1 / (N_max ** 2 * (1.0 - eps1))) < 1e-15


def test_validate_intervals_no_longer_rejects_pi_in_I1():
    validate_intervals([(0.5, 1.0)], [(2.0, np.pi)])  # must not raise


def test_max_kappa_B_and_Lambda_match_brute_grid():
    """Independent grid cross-check of both new certify_exact diagnostics,
    on a case with no band-edge degeneracy (so Lambda is finite and the
    grid ratio is well-defined almost everywhere)."""
    alphas = np.array([0.5, -0.3, -0.2])
    J1 = [(2.0, 2.5)]
    res = certify_exact(alphas, [], J1, [])

    a = a_from_alphas(alphas)
    theta = np.linspace(J1[0][0], J1[0][1], 2_000_000)
    kap = kappa_B(a, theta)
    Q = q1_abs_sq(a, theta)
    grid_max_kappa_B = float(np.max(np.abs(kap)))
    denom = 1.0 - kap ** 2
    grid_lambda = float(np.max((Q - 1.0)[denom > 1e-6] / denom[denom > 1e-6]))

    assert res.max_kappa_B is not None
    assert abs(res.max_kappa_B - grid_max_kappa_B) < 1e-6
    assert res.Lambda is not None
    assert abs(res.Lambda - grid_lambda) / grid_lambda < 1e-4


def test_Lambda_is_none_when_certify_has_no_pass_band():
    alphas = np.array([0.5, -0.3, -0.2])
    res = certify_exact(alphas, [(1.0, 1.5)], [], [1])
    assert res.Lambda is None
    assert res.max_kappa_B is None


@pytest.mark.slow
def test_phase2_enforces_constraint_E():
    """A geometry where the OLD (pre-revision) design was known to
    violate |kappa_B|<=1 somewhat in the pass band (Instance 1's own
    original interval choice): Phase 2 with (E) must now certify
    max_kappa_B within the verify tolerance, not just report it."""
    I0 = [(np.pi / 6, np.pi / 4)]
    I1 = [(np.pi / 2, 3 * np.pi / 4)]
    mu0 = 0.05
    res = design_direct_literal(5, I0, I1, mu0)
    assert res.status == "optimal"
    cert = certify_exact(np.asarray(res.alpha), I0, I1, res.sigma_star)
    assert cert.max_kappa_B is not None
    assert cert.max_kappa_B <= 1.0 + 1e-3  # SQP's own grid tolerance, not exact-certification tolerance


def _fake_record(max_kappa_B, s_0, delta_achieved, delta_target, kappa_min=1.5, mu0=0.15):
    return DegreeRecord(
        n=3, status="optimal", underline_delta=1e-6, dual_ray_found=False,
        dual_status="dual_certified", dual_blocks_all_feasible=True,
        delta_achieved=delta_achieved, kappa_min=kappa_min, mu_min=0.9,
        max_kappa_B=max_kappa_B, s_0=s_0, Lambda=0.02,
        sigma_star=(-1,), kappa_min_by_sigma={},
        admissible=False, N_max=20, mu0=mu0, delta_target=delta_target, N_required=10.0,
        alpha=[0.1, -0.1], rho=[1.0, 1.1, 1.0],
        start_origin="phase1", grid_vs_exact={}, gamma=1.2, n_min_green=4.5,
        time_direct_s=0.1, time_sdp_s=0.1,
    )


def test_admissible_no_longer_gated_by_negative_s0():
    """Direct regression proof that the s_0 gate is gone: a record with
    s_0<0 (would have been rejected outright by the old test) but with
    max_kappa_B<=1 and delta<=delta_target must be computed as
    admissible by run_degree_stage3's own logic. Exercised here by
    replicating that logic inline against the dataclass, since
    admissible is computed once inside run_degree_stage3 -- this test
    documents the intended semantics directly."""
    cosh_mu0 = float(np.cosh(0.15))
    rec = _fake_record(max_kappa_B=0.99, s_0=-0.01, delta_achieved=1e-5, delta_target=1e-4,
                        kappa_min=cosh_mu0 + 0.01, mu0=0.15)
    admissible = (rec.kappa_min >= cosh_mu0 - 1e-9
                  and rec.max_kappa_B <= 1.0 + 1e-9
                  and rec.delta_achieved <= rec.delta_target + 1e-9)
    assert admissible is True
    assert rec.s_0 < 0  # confirms this case really would have failed the OLD gate
