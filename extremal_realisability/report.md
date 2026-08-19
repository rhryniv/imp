# Do the extremal polynomials of the sharp bound yield realisable filters?

Geometry: `I1=[0,pi/4]`, `I0=[3pi/4,pi]`, `a=cos(pi/4)`, `b=cos(3pi/4)`,
`mu0=1`, `gamma=3.0571418389619963` (matches the brief's `3.0571418389619964`
to 16 digits). All of Steps 1-3 use `mpmath` at 60 decimal digits.

## Method notes

**Qhat_n** is built directly as a Chebyshev-in-`x` coefficient vector via
the affine substitution `l(x)=Ax+B` composed with `T_n` through the
3-term Chebyshev recurrence carried out entirely in that basis — the
monomial basis is never used, avoiding exactly the conditioning problem
this kind of high-degree construction is prone to.

**Spectral factorisation (p1)** follows the brief's suggested route:
build the degree-`2n` associated real polynomial, root it with
`mp.polyroots`, keep the `n` roots with `|z|>1`, normalise so `p1(1)=1`
exactly (forcing this, then verifying `|p1(e^{-i*vartheta})|^2=Q(vartheta)`
independently, rather than trusting the sign/normalisation blindly).

**Layer stripping (p2 and alpha_j) is fully explicit, not iterative.**
An initial attempt used a Newton solve for `alpha_1..alpha_n` (as the
brief's own fallback permits) and it worked, but a non-iterative route
is available and was used for the reported numbers: `p2`'s `n` roots are
exactly the equioscillation touch points where `Qhat_n=1` on `F_1`
(`l_k=cos(2*pi*k/n)`, `k=0..(n-1)/2`, mapped back through `x` to
`vartheta` — closed form from `T_n`'s own structure, no root-finding of a
new polynomial needed), normalised via `|p2(e^{-i*pi/4})|^2=delta_n`
(exact, since `Q(t)-1=delta_n` identically at `F_1`'s boundary). The
classical Schur-type downward recursion then peels `alpha_n,...,alpha_0`
one at a time, each step an *exact* polynomial division verified to have
zero remainder. The two methods' `alpha_j` agree to the full precision
tested; the explicit route is reported below and is what a reviewer
would expect for "layer stripping."

## Output 4 — verification log

All residuals below are the *maximum* absolute deviation found, in
60-digit `mpmath` arithmetic (grids of 400-2000 points as appropriate).

| n | delta_n reldiff vs brief | 5 properties (max abs) | rho=min\|z_l\| | p1 vs Q check | p1(1)-1 | Schur max residual | forward-map check |
|---|---|---|---|---|---|---|---|
| 1 | 1.6e-16 | ~1.6e-61 | 4.2360311 | 1.2e-60 | 0 | 7.8e-62 | 3.1e-61 |
| 3 | 2.8e-17 | ~1.2e-60 | 2.7860743 | 1.2e-60 | 0 | 9.7e-63 | 1.6e-60 |
| 5 | 2.0e-16 | ~6.2e-61 | 2.2177671 | 1.2e-60 | 0 | 1.7e-59 | 1.2e-59 |
| 7 | 3.4e-11* | ~1.9e-60 | 1.9244829 | 1.9e-60 | 0 | 2.0e-57 | 2.6e-57 |
| 9 | 7.5e-11* | ~1.2e-60 | 1.7450131 | 3.7e-60 | 0 | 2.3e-56 | 3.6e-56 |

(*n=7,9's "reldiff vs brief" is limited by the brief quoting `delta_n`
to only 10 significant figures for those two rows; our own value is
internally consistent to 60 digits — see `geometry.py`'s output.)

All **five properties of Section 2 hold** (max deviation `~1e-60`, i.e.
exact to 60-digit precision): `Qhat_n>=1` on `[-1,1]`; `Qhat_n<=1+delta_n`
on `F_1`; `Qhat_n>=cosh^2(mu0)` on `F_0`; `Qhat_n(1)=1`; `deg=n` with
nonzero leading coefficient. `rho=min|z_l|` decreases monotonically
toward 1 as `n` grows (4.24 -> 2.79 -> 2.22 -> 1.92 -> 1.75), exactly the
conditioning degradation the brief anticipates — 60 digits was ample
margin throughout (no precision failures at any `n`). The Fejer-Riesz
check `|p1(e^{-i*vartheta})|^2=Q(vartheta)` holds to `~1e-60` on a
201-point grid for every `n`; `p1(1)=1` exactly, as predicted. The Schur
downward recursion's own internal consistency checks (each step's
degree-cancellation and exact-division residuals, plus
`cosh^2(alpha_0)-sinh^2(alpha_0)-1=0`) are all at the 60-digit rounding
floor, and the forward recursion, run on the recovered `alpha_j`,
reproduces `p1` to `3e-61`-`4e-56` (growing mildly with `n`, tracking the
conditioning trend, never a concern).

## Output 2 — contrasts and impedances (4 s.f.)

| n | alpha_j | rho_j = exp(alpha_0+...+alpha_{j-1}) |
|---|---|---|
| 1 | 0.5306, -0.5306 | 1.700 |
| 3 | 0.1753, -0.4283, 0.4283, -0.1753 | 1.192, 0.7765, 1.192 |
| 5 | 0.05587, -0.2239, 0.3977, -0.3977, 0.2239, -0.05587 | 1.057, 0.8453, 1.258, 0.8453, 1.057 |
| 7 | 0.01774, -0.0998, 0.2532, -0.3838, 0.3838, -0.2532, 0.0998, -0.01774 | 1.018, 0.9212, 1.187, 0.8084, 1.187, 0.9212, 1.018 |
| 9 | 0.005604, -0.04079, 0.1358, -0.2739, 0.3777, -0.3777, 0.2739, -0.1358, 0.04079, -0.005604 | 1.006, 0.9654, 1.106, 0.8409, 1.227, 0.8409, 1.106, 0.9654, 1.006 |

(`alpha_j` are exactly antisymmetric, `alpha_j=-alpha_{n-j}`, and `rho_j`
symmetric about the centre — a consequence of the construction, not
imposed. There is a genuine, harmless sign ambiguity noted during
development: flipping the overall sign of every `alpha_j` reproduces the
identical `p1` — e.g. at `n=1` both `(alpha_0,alpha_1)=(+0.5306,-0.5306)`
and `(-0.5306,+0.5306)` give the same block. This is a left-right mirror
of the medium and does not affect `kappa_B`, `kappa_min`, or `mu_eff`.)

## Output 1 & 3 — the realisability test

For each `n`, `L_best` is chosen within the brief's own `[2n,6n]` range
(extended when `kappa_min` was still rising at the top of that range, as
instructed); a *supplementary* wide check over `[2n,12n]` is also
reported to confirm the qualitative picture is not an artifact of the
swept range.

| n | delta_n | L_best | kappa_min | mu_eff | mu_eff>=1? | (E) holds? | sign on I_0 |
|---|---|---|---|---|---|---|---|
| 1 | 2.3696e-01 | 2 | 1.23464 | 0.67230 | **No** | Yes | constant (-) |
| 3 | 5.7424e-04 | 16 (range extended to 21) | 0.0017076 | undefined | No | No | **not constant** |
| 5 | 1.2699e-06 | 20 (range extended to 35) | 0.0026676 | undefined | No | No | **not constant** |
| 7 | 2.8079e-09 | 48 (range extended to 49) | 0.0049934 | undefined | No | Yes | **not constant** |
| 9 | 6.2085e-12 | 48 (range extended to 63) | 0.0046411 | undefined | No | Yes | **not constant** |

"`mu_eff` undefined" means `kappa_min<1` — the design does not even open
a band gap covering all of `I0`, let alone one deep enough for `mu0=1`.

**The full `kappa_min` vs `L` profiles** (see figure
`kappa_min_vs_L.pdf`/`.png`): for `n=1`, `kappa_min` is `1.235` at `L=2`,
collapses to `~0` at every odd `L` (a resonance closing the gap
entirely), and reaches only `0.219` at `L=4` before decaying further —
clearly non-monotone in `L`, as expected. For `n=3,5,7,9`, the profile
oscillates rapidly between values indistinguishable from `0` (`~1e-60`,
i.e. `kappa_B` passes exactly through the band edge or beyond) and small
positive peaks that never exceed `~0.005`, across the *entire* swept
range — there is no trend toward `1` anywhere.

**Supplementary wide check, `L=2n..12n`:** `n=1` finds constant sign at
5 of 11 tested `L` (`L=2,4` genuine, `L=3,5,7` the near-zero resonances);
`n=3` finds it at exactly 1 of 31 (`L=7`, itself a near-zero degenerate
case, `kappa_min~7e-61`, not a real gap); `n=5,7,9` find **zero** constant-sign
`L` values out of 51, 71, and 91 tested respectively. This was checked
well beyond the brief's own `6n` ceiling specifically because the
`kappa_min`-still-rising trigger kept firing — extending never revealed
a genuine gap for `n>=3`.

## Output 5 — which outcome occurred, and why

**Outcome 2 (relaxation degrades), and more sharply than a "shortfall
that grows with n":** already at `n=1` the relaxed optimum is not
realised (`mu_eff=0.672<1`, consistent with the brief's own `1.8034233`
gap ratio), and from `n=3` on the family doesn't produce a *bounded* gap
at all — `kappa_B` fails to hold a constant sign anywhere in `I0` for
essentially every tested period (`0` hits out of 51-91 for `n=5,7,9`
even after extending far past `6n`), so `mu_eff` is not merely small, it
is undefined.

**Mechanism** (confirmed directly, not assumed): `kappa_B(vartheta;L) =
sqrt(Q(vartheta)) * cos(L*vartheta/2 + psi(vartheta))`, where `psi` is
`p1`'s own phase. `psi`'s intrinsic variation across `I0` is modest and
actually *grows slowly* with `n` (`0.045*pi` at `n=1` up to `0.208*pi` at
`n=9`) — not the driver. The driver is the `L*vartheta/2` term: `L` must
be at least `2n` (one full period per block), so its phase sweep across
`I0`'s fixed width `pi/4` grows linearly with `n`, guaranteeing multiple
sign crossings once `L` is large enough — and `L` is *forced* to be large
once `n` is, since `L>=2n`. This is exactly the mechanism you flagged:
**a cosine polynomial of growing degree `n`, evaluated over the fixed
angular window `I0`, necessarily sweeps back and forth across
`kappa_B in (-1,1)` many times once `n` (hence the minimum admissible
`L`) is large enough — large-`n` designs on a fixed-width stop band are
generically self-defeating for this construction**, independent of how
good `Qhat_n` itself is as a magnitude-relaxation optimum.

## Files

`geometry.py`, `qhat.py` (Chebyshev-basis `Qhat_n` construction),
`spectral.py` (Fejer-Riesz), `schur.py` (explicit `p2` + downward
peeling — the reported route), `layers.py` (the Newton-based alternative,
kept as a cross-check), `run_all.py` (full pipeline + `L`-sweep),
`plot_profiles.py` (figure), `run_all_output.txt` (full verification
log), `run_all_results.json`.
