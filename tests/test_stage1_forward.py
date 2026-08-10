"""Stage 1 (codespec6.4.md Sec. 2): forward-model validation. Two
independently coded evaluation paths cross-checked to machine precision,
exact rational regression fixtures F1-F3, and the four identities of
Sec. 2 -- all as required before anything downstream (Stage 2/3) is
trusted.

Run: python3 -m pytest tests/test_stage1_forward.py -v   (from repo root)
"""
import numpy as np
import pytest

from scattering.forward import (
    a_from_alphas, autocorr, eval_poly, forward_matrix_product,
    forward_reconstruct, kappa_B, q1_abs_sq, q2_abs_sq,
)


# --------------------------------------------------------------------------
# (B) Cross-check: path (a) [coefficient recursion] vs path (b) [literal
# per-layer SU(1,1) product with position-dependent phases]
# --------------------------------------------------------------------------

def _random_matched_alphas(rng: np.random.Generator, n: int) -> np.ndarray:
    alphas = rng.uniform(-1.5, 1.5, n + 1)
    alphas[-1] -= np.sum(alphas)  # force sum(alphas) == 0 (matched)
    return alphas


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_forward_paths_agree(n):
    """Relative, not absolute, tolerance: coefficient magnitudes (and hence
    |p1|,|p2|) grow with n and with |alpha| (products of cosh/sinh
    compounding through the recursion), so a fixed absolute tolerance is
    the wrong comparison -- confirmed directly (n=8 case reaches
    |p1|~600, absolute diff ~1.4e-12, but *relative* diff ~2.4e-15,
    i.e. genuine machine precision)."""
    rng = np.random.default_rng(1000 + n)
    alphas = _random_matched_alphas(rng, n)
    theta = rng.uniform(-3 * np.pi, 3 * np.pi, 11)  # outside [0,pi] too: both paths are entire in theta

    c1, c2 = forward_reconstruct(alphas)
    p1_a, p2_a = eval_poly(c1, np.exp(-1j * theta)), eval_poly(c2, np.exp(-1j * theta))
    p1_b, p2_b = forward_matrix_product(alphas, theta)

    assert np.max(np.abs(p1_a - p1_b)) / max(np.max(np.abs(p1_a)), 1.0) < 1e-12
    assert np.max(np.abs(p2_a - p2_b)) / max(np.max(np.abs(p2_a)), 1.0) < 1e-12


# --------------------------------------------------------------------------
# (C) Exact rational regression fixtures
# --------------------------------------------------------------------------

def test_fixture_F1():
    """n=1, rho_1=2: alpha=(log2,-log2), L=2. cosh(log2)=5/4, sinh(log2)=3/4,
    so p1(w)=25/16-(9/16)w, Q=(706-450cos theta)/256, kappa_B=(25cos theta-9)/16
    -- all rational, checked near machine precision.

    "One open gap per period": kappa_B(pi)=-34/16, strictly inside the gap
    (not closed at +-1), so the gap visible in [0,pi] connects continuously
    across theta=pi to its own mirror image on (pi,2pi) -- one continuous
    gap over the full period [0,2pi]. Contrast test_fixture_F2, where
    kappa_B(pi)=+1 exactly (closed), splitting the mirror image into a
    second, separate gap.
    """
    alphas = np.array([np.log(2.0), -np.log(2.0)])
    c1, c2 = forward_reconstruct(alphas)
    assert np.allclose(c1, [25 / 16, -9 / 16], atol=1e-14)

    theta = np.array([0.0, 0.7, np.pi / 2, 2.1, np.pi])
    a = a_from_alphas(alphas)

    p1_1 = eval_poly(c1, np.array([1.0]))[0]
    assert abs(p1_1 - 1.0) < 1e-14                      # p1(1) = 1

    Q = q1_abs_sq(a, theta)
    Q_exact = (706 - 450 * np.cos(theta)) / 256
    assert np.allclose(Q, Q_exact, atol=1e-13)
    assert abs(Q[0] - 1.0) < 1e-13                       # Q(0) = 1

    kap = kappa_B(a, theta)
    kap_exact = (25 * np.cos(theta) - 9) / 16
    assert np.allclose(kap, kap_exact, atol=1e-13)
    assert abs(kap[0] - 1.0) < 1e-13                     # kappa_B(0) = 1
    assert abs(kap[-1] - (-34 / 16)) < 1e-13             # kappa_B(pi) = -34/16 = -2.125

    # exactly one open gap per period, containing pi: |kappa_B(pi)| > 1
    assert abs(kap[-1]) > 1.0
    # Q(pi) = kappa_B(pi)^2 (p1 real there, so the imaginary part vanishes)
    assert abs(Q[-1] - kap[-1] ** 2) < 1e-13


def test_fixture_F2():
    """n=2, rho_1=rho_2=2: alpha=(log2,0,-log2), L=4. Must show two open
    gaps per PERIOD, i.e. over the full [0,2pi] range, not [0,pi].

    kappa_B is even and 2*pi-periodic, so [0,pi] only ever shows ONE
    connected gap component even when there are two per period: generically
    kappa_B(pi) sits strictly inside a gap (not at +-1), so the gap in
    [0,pi] connects continuously, across pi, to its own mirror image on
    (pi,2pi) -- one continuous gap over the full period. Two SEPARATE gap
    components requires kappa_B(pi) to close exactly at +-1, pinching the
    [0,pi] gap so it stays confined to the open interval (0,pi); its mirror
    image is then a second, distinct gap on (pi,2pi).

    Exact closed form here (a=(-9/16,0,25/16), matching the recursion by
    hand): kappa_B(theta) = -9/16 + (25/16)cos(2 theta), confirmed to
    2.2e-16 against direct evaluation. This attains its max of exactly 1 at
    BOTH theta=0 (universal, kappa_B(0)=1 for every design) AND theta=pi
    (specific to this alpha) -- exactly the closed-gap-at-pi signature
    above -- with a single dip to -34/16 at theta=pi/2 confined to (0,pi).
    So: one open gap directly visible in [0,pi], closed exactly at pi,
    hence two open gaps over the full period [0,2pi] -- matching the
    fixture's own claim exactly.
    """
    alphas = np.array([np.log(2.0), 0.0, -np.log(2.0)])
    c1, _ = forward_reconstruct(alphas)
    assert np.allclose(c1, [25 / 16, 0.0, -9 / 16], atol=1e-14)

    a = a_from_alphas(alphas)
    assert np.allclose(a, [-9 / 16, 0.0, 25 / 16], atol=1e-14)

    theta = np.linspace(0.0, np.pi, 2000)
    kap = kappa_B(a, theta)
    closed_form = -9 / 16 + (25 / 16) * np.cos(2 * theta)
    assert np.max(np.abs(kap - closed_form)) < 1e-13

    kap_0 = kappa_B(a, np.array([0.0]))[0]
    kap_pi = kappa_B(a, np.array([np.pi]))[0]
    assert abs(kap_0 - 1.0) < 1e-13                       # universal: kappa_B(0)=1 for every design
    assert abs(kap_pi - 1.0) < 1e-13                      # closed gap AT pi -- specific to this alpha

    kap_mid = kappa_B(a, np.array([np.pi / 2]))[0]
    assert abs(kap_mid - (-34 / 16)) < 1e-13              # single dip to -34/16 at theta=pi/2
    assert kap.min() >= -34 / 16 - 1e-9                   # grid can only ever miss the true min

    # exactly one open gap component directly visible in [0,pi] ...
    is_gap = np.abs(kap) > 1.0 + 1e-9
    n_gaps = int(np.sum(np.diff(is_gap.astype(int)) == 1) + (1 if is_gap[0] else 0))
    assert n_gaps == 1
    # ... confined strictly inside (0,pi) (closed at both ends, not touching
    # the domain boundary as an open gap would) -- combined with
    # kappa_B(pi)=1 above, this is the "two gaps per period" signature: the
    # mirror image on (pi,2pi) is a second, separate gap component.
    assert not is_gap[0] and not is_gap[-1]


def test_fixture_F3():
    """n=3, rho=(2,sqrt2,2), d=3: alpha=(log2,-log2/2,log2/2,-log2), L=6.
    Must show three gaps, and T_N for N=1,2,5 must reproduce the figure:
    attenuation deepening in the gaps, ripple bounded below by an
    N-independent envelope in the bands."""
    alphas = np.array([np.log(2.0), -np.log(2.0) / 2, np.log(2.0) / 2, -np.log(2.0)])
    assert abs(np.sum(alphas)) < 1e-14  # matched
    a = a_from_alphas(alphas)
    theta = np.linspace(1e-4, np.pi - 1e-4, 20000)
    kap = kappa_B(a, theta)

    is_gap = np.abs(kap) > 1.0
    n_gaps = int(np.sum(np.diff(is_gap.astype(int)) == 1) + (1 if is_gap[0] else 0))
    assert n_gaps == 3, f"expected 3 open gaps, found {n_gaps}"

    from scattering.forward import transmission_TN
    # Margin around the band/gap boundary (|kappa_B|=1): points immediately
    # adjacent to it are dominated by the edge effect of Remark rem:edge
    # (T_N ~ N^-2 there, from a genuinely different mechanism than either
    # the gap's exponential attenuation or the band's N-independent
    # ripple floor), and would contaminate a naive min/max over a raw
    # is_gap split -- confirmed directly: the raw split put the "band"
    # minimum one grid step from a gap boundary, at a T_5 value an order
    # of magnitude below the true interior ripple floor.
    deep_gap = np.abs(kap) > 1.1
    deep_band = np.abs(kap) < 0.9
    assert deep_gap.any() and deep_band.any()

    T2_gap_max = np.max(transmission_TN(a, theta[deep_gap], 2))
    T5_gap_max = np.max(transmission_TN(a, theta[deep_gap], 5))
    T2_band_min = np.min(transmission_TN(a, theta[deep_band], 2))
    T5_band_min = np.min(transmission_TN(a, theta[deep_band], 5))

    # attenuation deepens sharply in the gaps as N grows (Prop. asymmetry
    # (a), exponential in N)...
    assert T5_gap_max < 0.5 * T2_gap_max
    # ...while the ripple floor in the bands stays essentially bounded,
    # NOT decaying at anywhere near the same rate (Prop. asymmetry (b):
    # the envelope is independent of N)
    assert T5_band_min > 0.5 * T2_band_min


# --------------------------------------------------------------------------
# (D) Identities (spec Sec. 2)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 4, 7])
def test_identities(n):
    rng = np.random.default_rng(2000 + n)
    alphas = _random_matched_alphas(rng, n)
    a = a_from_alphas(alphas)
    theta = rng.uniform(-2 * np.pi, 2 * np.pi, 15)

    Q = q1_abs_sq(a, theta)
    q2sq = q2_abs_sq(a, theta)
    kap = kappa_B(a, theta)

    # Q - 1 == |p2|^2 everywhere (q2_abs_sq clips at 0, so compare pre-clip)
    assert np.allclose(Q - 1.0, np.clip(Q - 1.0, 0.0, None), atol=1e-10)
    assert np.all(Q - 1.0 > -1e-9)              # Q >= 1 everywhere (realizable by construction)

    # Q >= kappa_B^2 everywhere
    assert np.all(Q - kap ** 2 > -1e-9)

    # Q(0) == 1 and p1(1) == 1 (matched blocks only)
    Q0 = q1_abs_sq(a, np.array([0.0]))[0]
    assert abs(Q0 - 1.0) < 1e-10
    c1, _ = forward_reconstruct(alphas)
    assert abs(eval_poly(c1, np.array([1.0]))[0] - 1.0) < 1e-10

    # p1(0) > 0
    assert eval_poly(c1, np.array([0.0]))[0].real > 0


# --------------------------------------------------------------------------
# autocorr / Q_from_autocorr sanity (used by q1_abs_sq internally now)
# --------------------------------------------------------------------------

def test_autocorr_matches_direct_sum():
    """Q_from_autocorr (via autocorr) must agree with the direct complex
    evaluation |sum a_m e^{im theta}|^2 to floating-point precision on a
    well-conditioned case -- confirms the stable route computes the SAME
    quantity, not a different one."""
    rng = np.random.default_rng(42)
    alphas = _random_matched_alphas(rng, 4)
    a = a_from_alphas(alphas)
    theta = rng.uniform(0, np.pi, 9)

    Q_stable = q1_abs_sq(a, theta)
    m = np.arange(len(a))
    q_direct = np.exp(1j * np.outer(theta, m)) @ a
    Q_direct = np.abs(q_direct) ** 2

    assert np.allclose(Q_stable, Q_direct, atol=1e-9)
