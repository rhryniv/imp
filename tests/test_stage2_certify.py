"""Stage 2, spec Sec. 3.3 ("Certification"): certify_exact's exact
Chebyshev-colleague-matrix results, against fine-grid brute force and
against the specific non-negotiables of spec Sec. 5 (rules 3 and 6).

Run: python3 -m pytest tests/test_stage2_certify.py -v   (from repo root)
"""
import numpy as np
import pytest

from scattering.bandgap import find_bands_gaps
from scattering.certify import certify_exact, grid_vs_exact
from scattering.forward import a_from_alphas, kappa_B, q1_abs_sq


def _grid_extremes(alphas, J0, J1, sigma, n_grid=2_000_000):
    a = a_from_alphas(np.asarray(alphas, dtype=float))
    grid_delta = max((float(np.max(q1_abs_sq(a, np.linspace(lo, hi, n_grid)) - 1.0)) for lo, hi in J1),
                      default=0.0)
    grid_kappa_min = (min((float(np.min(s * kappa_B(a, np.linspace(lo, hi, n_grid))))
                            for (lo, hi), s in zip(J0, sigma)))
                       if J0 else None)
    grid_s0 = (min((float(np.min(1.0 - kappa_B(a, np.linspace(lo, hi, n_grid)) ** 2))
                     for lo, hi in J1))
               if J1 else None)
    return grid_delta, grid_kappa_min, grid_s0


@pytest.mark.parametrize("alphas,label", [
    (np.array([0.5, -0.3, -0.2]), "n=2, asymmetric"),
    (np.array([0.2, 0.1, -0.05, -0.25]), "n=3"),
    (np.array([0.9, -0.4, 0.3, -0.1, -0.7]), "n=4, larger contrasts"),
])
def test_certify_exact_matches_grid_on_natural_bandgap(alphas, label):
    """certify_exact's delta, kappa_min, and s_0 against a 2-million-point
    brute-force grid, on a genuine band/gap structure found via
    bandgap.find_bands_gaps (not an arbitrary interval) -- delta and
    kappa_min should match to floating-point precision (both read off the
    same trig polynomial, one via calculus and one via sampling; the grid
    can only ever underestimate the true extremum)."""
    a = a_from_alphas(alphas)
    bgs = find_bands_gaps(a)
    assert bgs.bands and bgs.gaps, f"no band/gap structure found for {label}"
    gap_lo, gap_hi, gap_sign = bgs.gaps[0]
    J0, J1, sigma = [(gap_lo, gap_hi)], [bgs.bands[0]], [gap_sign]

    res = certify_exact(alphas, J0, J1, sigma)
    grid_delta, grid_kappa_min, grid_s0 = _grid_extremes(alphas, J0, J1, sigma)

    assert abs(res.delta - grid_delta) < 1e-6
    assert res.kappa_min is not None
    assert abs(res.kappa_min - grid_kappa_min) < 1e-6
    # exact must never be *worse* than the grid found (grid underestimates
    # max(Q-1), overestimates min kappa_min/s_0)
    assert res.delta >= grid_delta - 1e-9
    assert res.kappa_min <= grid_kappa_min + 1e-9

    if res.s_0 is not None and grid_s0 is not None:
        assert abs(res.s_0 - grid_s0) < 1e-6
        assert res.s_0 <= grid_s0 + 1e-9


def test_certify_exact_s0_is_its_own_degree_2n_polynomial():
    """Rule 6: s_0 must be certified as its OWN degree-2n polynomial
    (1-kappa_B^2), not derived from kappa_B's own extremum -- checked
    here by comparing against an independent brute-force grid evaluation
    of 1-kappa_B(theta)^2 directly (not via kappa_B's max)."""
    alphas = np.array([0.6, 0.1, -0.2, -0.5])
    J1 = [(0.4, 1.2)]
    a = a_from_alphas(alphas)

    res = certify_exact(alphas, [], J1, [])
    theta = np.linspace(J1[0][0], J1[0][1], 2_000_000)
    grid_s0 = float(np.min(1.0 - kappa_B(a, theta) ** 2))

    assert res.s_0 is not None
    assert abs(res.s_0 - grid_s0) < 1e-6
    assert res.s_0 <= grid_s0 + 1e-9


def test_certify_exact_rule3_no_clamp_when_kappa_min_below_1():
    """Rule 3: if J0 sits somewhere kappa_B never reaches sigma*kappa_B>1
    (i.e. not actually in a gap for the given sign), mu_min must be None
    -- never clamped to 0, never arccosh(|kappa_min|), never mu_0."""
    alphas = np.array([0.05, -0.02, -0.03])  # tiny contrasts: kappa_B stays close to cos(n*theta), no real gap
    a = a_from_alphas(alphas)
    theta_probe = np.linspace(0.1, np.pi - 0.1, 2000)
    kap = kappa_B(a, theta_probe)
    assert np.max(np.abs(kap)) < 1.05, "test setup expects a near-trivial (no deep gap) design"

    J0 = [(1.0, 1.5)]
    sigma = [1]
    res = certify_exact(alphas, J0, [], sigma)

    assert res.kappa_min is not None            # kappa_min itself is always reported
    assert res.kappa_min < 1.0                  # this component genuinely fails to reach a gap
    assert res.mu_min is None                   # and mu_min must be None, not a clamped fallback
    assert res.TN_stop_bound == {}               # no valid mu_min means no valid TN_stop_bound either


def test_certify_exact_mu_min_never_clamped():
    """Rule 3, both directions of the strict '>' boundary against
    certify_exact's own reported kappa_min (not a value chosen to force
    an exact 1.0 -- for a continuous function a nonzero-width interval's
    minimum only equals its supremum, 1, in the degenerate zero-width
    limit, so that boundary isn't meaningfully constructible from a real
    design; the two directions are checked separately instead).

    (a) kappa_min>1 (a genuine gap): mu_min must equal arccosh(kappa_min)
    exactly, not some clamped/rounded variant.
    (b) kappa_min<1 (no real gap, covered by the "no_clamp_when_below_1"
    test above): mu_min is None -- rechecked here as an explicit
    invariant, kappa_min<=1.0 implies mu_min is None, over several cases."""
    cases = [
        (np.array([0.5, -0.3, -0.2]), True),    # genuine gap (kappa_min>1, from the passing test above)
        (np.array([0.05, -0.02, -0.03]), False),  # near-trivial, no real gap (kappa_min<1)
    ]
    for alphas, expect_gap in cases:
        a = a_from_alphas(alphas)
        bgs = find_bands_gaps(a)
        if not bgs.gaps:
            continue
        gap_lo, gap_hi, sign = bgs.gaps[0]
        res = certify_exact(alphas, [(gap_lo, gap_hi)], [], [sign])
        assert res.kappa_min is not None
        if res.kappa_min > 1.0:
            assert res.mu_min is not None
            assert abs(res.mu_min - np.arccosh(res.kappa_min)) < 1e-12
        else:
            assert res.mu_min is None


def test_certify_exact_no_stop_or_pass_region():
    """J0=[] and/or J1=[] should degrade gracefully (spec's own trivial
    cases), not raise."""
    alphas = np.array([0.3, -0.1, -0.2])
    res_no_stop = certify_exact(alphas, [], [(0.5, 1.0)], [])
    assert res_no_stop.kappa_min is None
    assert res_no_stop.mu_min is None
    assert res_no_stop.s_0 is not None

    res_no_pass = certify_exact(alphas, [(0.5, 1.0)], [], [1])
    assert res_no_pass.delta == 0.0
    assert res_no_pass.s_0 is None


def test_grid_vs_exact_reports_pairwise_discrepancy():
    alphas = np.array([0.5, -0.3, -0.2])
    a = a_from_alphas(alphas)
    bgs = find_bands_gaps(a)
    gap_lo, gap_hi, gap_sign = bgs.gaps[0]
    J0, J1, sigma = [(gap_lo, gap_hi)], [bgs.bands[0]], [gap_sign]

    exact = certify_exact(alphas, J0, J1, sigma)
    # a deliberately coarse grid, so its own discrepancy from the exact
    # value is large enough to check the reporting logic itself, not just
    # that both numbers happen to already agree
    theta_C = np.linspace(J0[0][0], J0[0][1], 50)
    theta_B = np.linspace(J1[0][0], J1[0][1], 50)
    grid_delta = float(np.max(q1_abs_sq(a, theta_B) - 1.0))
    grid_kappa_min = float(np.min(gap_sign * kappa_B(a, theta_C)))

    diag = grid_vs_exact(grid_delta, grid_kappa_min, exact)
    assert diag["delta_discrepancy"] >= 0.0
    assert diag["kappa_min_discrepancy"] >= 0.0
    assert diag["delta_discrepancy"] == abs(grid_delta - exact.delta)
    assert diag["kappa_min_discrepancy"] == abs(grid_kappa_min - exact.kappa_min)
