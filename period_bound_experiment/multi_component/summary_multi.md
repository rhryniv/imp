# Two multi-component geometries — summary

## 1. Gate result

Re-ran the Sec 6.2 geometry (I1=[0,pi/4], I0=[5pi/6,pi], single component
each) at n=3, d=1, sigma=+1: **delta = 4.6722221733e-04** against target
4.6722e-4 — **PASS**. Proceeded to the full 56-case sweep.

## 2. The optima found

Feasibility was much sparser than the single/two-component-stop period
run: **only 6 of 56 cases were feasible** (11%), against 6/66 (9%) in
the earlier single-geometry sweep — comparably sparse, but here every
feasible case is at d=1 or d=3, never higher.

| Geometry | n | d | L | sigma | delta | best_start_type | n_converged/total |
|---|---|---|---|---|---|---|---|
| A | 3 | 1 | 4 | (+1,+1) | 3.987614e-01 | random | 20/250 |
| A | 3 | 1 | 4 | (-1,+1) | 1.735323e-01 | random | 247/250 |
| A | 3 | 3 | 6 | (+1,-1) | 9.625639e-03 | random | 205/251 |
| A | 5 | 1 | 6 | (+1,-1) | 4.679160e-05 | random | 146/251 |
| B | 3 | 1 | 4 | (-1) | 1.123691e+00 | random | 232/300 |
| B | 5 | 1 | 6 | (+1) | 1.694513e+00 | random | 255/300 |

**Best per geometry**: Geometry A, n=5, d=1, sigma=(+1,-1), delta=4.68e-5
(the only n=5 result and by far the smallest delta found in either
geometry). Geometry B, n=3, d=1, sigma=(-1), delta=1.124 (its only
competitor, B's n=5 result, is worse at 1.695) — every feasible Geometry
B design has delta of order 1, i.e. far from a good filter; Geometry B
is a substantially harder design problem within this search budget than
Geometry A.

Every remaining d value in both geometries — d=5,...,13 for A (both n),
d=2,...,7 for B (both n) — returned **no feasible point within the
search budget** for either sign. This is stated as "not found," not as
proof; see Sec 7 trap note and the caveats in Sec 6 below.

### delta vs (d, sigma), per geometry and n

"—" = infeasible (no start converged to a fine-grid-feasible point
within budget). Geometry A's sigma column shows (sigma_1, sigma_2), with
sigma_2 forced by parity ((-1)^(L/2)); Geometry B's sigma is the single
free sigma_1.

**Geometry A, n=3** (sigma_2 forced: d=1→+1, d=3→-1, d=5→+1, d=7→-1, d=9→+1, d=11→-1, d=13→+1)

| d | L | sigma_1=+1 | sigma_1=-1 |
|---|---|---|---|
| 1 | 4 | 3.987614e-01 | 1.735323e-01 |
| 3 | 6 | 9.625639e-03 | — |
| 5 | 8 | — | — |
| 7 | 10 | — | — |
| 9 | 12 | — | — |
| 11 | 14 | — | — |
| 13 | 16 | — | — |

**Geometry A, n=5** (sigma_2 forced: d=1→-1, d=3→+1, d=5→-1, d=7→+1, d=9→-1, d=11→+1, d=13→-1)

| d | L | sigma_1=+1 | sigma_1=-1 |
|---|---|---|---|
| 1 | 6 | 4.679160e-05 | — |
| 3 | 8 | — | — |
| 5 | 10 | — | — |
| 7 | 12 | — | — |
| 9 | 14 | — | — |
| 11 | 16 | — | — |
| 13 | 18 | — | — |

**Geometry B, n=3**

| d | L | sigma=+1 | sigma=-1 |
|---|---|---|---|
| 1 | 4 | — | 1.123691e+00 |
| 2 | 5 | — | — |
| 3 | 6 | — | — |
| 4 | 7 | — | — |
| 5 | 8 | — | — |
| 6 | 9 | — | — |
| 7 | 10 | — | — |

**Geometry B, n=5**

| d | L | sigma=+1 | sigma=-1 |
|---|---|---|---|
| 1 | 6 | 1.694513e+00 | — |
| 2 | 7 | — | — |
| 3 | 8 | — | — |
| 4 | 9 | — | — |
| 5 | 10 | — | — |
| 6 | 11 | — | — |
| 7 | 12 | — | — |

## 3. P1 — alternation count

Prediction: $p=(n+3)/2-m_0$ equioscillation points on $I_1$, with $m_0$
the geometry's structural stop-component count (A: 2, B's own analogue
using the single stop component, $m_0=1$, giving the Sec 6.2 form
$(n+1)/2$). Active-point counts were recomputed with proper clustering
(grouping contiguous fine-grid nodes within tolerance into one point —
the raw per-grid-point count used during the sweep drastically
overcounts near-flat peaks, e.g. one case initially showed "45 active
points" that clustering resolves to 3 genuine ones, verified by direct
inspection of local maxima and the interval endpoint).

| Geometry | n | m0 | predicted p=(n+3)/2-m0 | observed p (total over I1) | match? |
|---|---|---|---|---|---|
| A | 3 | 2 | 1 | 1 (both d=1 cases) and 1 (d=3 case) | **confirmed** |
| A | 5 | 2 | 2 | 3 | **refuted** (one extra) |
| B | 3 | 1 | 2 | 2 (1+1 across the two I1 components) | **confirmed** |
| B | 5 | 1 | 3 | 1 (0+1 across the two I1 components) | **refuted** (two fewer) |

**P1 is confirmed at n=3 in both geometries, and refuted at n=5 in
both.** The n=3 confirmations are on genuinely good designs (delta of
order 1e-1 to 1e-2, hundreds of converged starts) so carry real weight.
The n=5 results are more mixed in reliability: Geometry A's n=5 case
(delta=4.68e-5, 146/251 converged — a good design) shows one MORE active
point than predicted, traced to the right endpoint of I1 (theta=pi/4)
becoming an additional active point of (B) beyond the interior
equioscillation structure (confirmed by direct inspection: residual
Q(pi/4)-1-delta = -4.7e-10, i.e. genuinely touching, not a grid
artifact) — a mechanism the simple alternation-count formula does not
account for. Geometry B's n=5 case (delta=1.69, the worst design found
in either geometry, though still with 255/300 converged) shows two
FEWER active points than predicted; given its very large delta this
design does not obviously sit in the same "near-optimal equioscillating"
regime the formula describes, so this refutation carries less
evidential weight than A's.

## 4. P2 — symmetry

$\max_j|\alpha_j+\alpha_{n-j}|$ (antisym. residual) and
$\max_j|\alpha_j-\alpha_{n-j}|$ (sym. residual), full (n+1)-length alpha:

| Case | antisym. residual | sym. residual | classification |
|---|---|---|---|
| A n=3 d=1 sigma=(+1,+1) | 5.09e-09 | 2.397 | **antisymmetric** |
| A n=3 d=1 sigma=(-1,+1) | 0.549 | 0.921 | **neither** |
| A n=3 d=3 sigma=(+1,-1) | 5.39e-08 | 1.718 | **antisymmetric** |
| A n=5 d=1 sigma=(+1,-1) | 1.85e-07 | 1.685 | **antisymmetric** |
| B n=3 d=1 sigma=(-1) | 0.846 | 0.494 | **neither** |
| B n=5 d=1 sigma=(+1) | 8.55e-05 | 1.190 | **antisymmetric** |

**Most interesting single result of this experiment, as anticipated**:
4 of 6 optima are antisymmetric to ~1e-5 or better (including one in
Geometry B, where the exclusion no longer formally applies but
antisymmetry is evidently still often optimal), while 2 are **neither**
symmetric nor antisymmetric — a genuinely broken-symmetry optimum
(A n=3 d=1 sigma=(-1,+1), and B's only n=3 result). No case came out
purely symmetric: Geometry B's 50 alpha_j=alpha_{n-j} starts were tried
in every B case (per spec Sec 4) but never produced the winning point in
the full run (they did win one B cell in an unrelated smoke test that
consumed the RNG stream differently — not part of the reported run).
So: when antisymmetry breaks, it breaks into a genuinely asymmetric
configuration, not into the symmetric one.

## 5. P3 — period bound

| Geometry | n | largest feasible d found | cap | cap/largest ratio |
|---|---|---|---|---|
| A | 3 | 3 | 13.33 | 4.4x |
| A | 5 | 1 | 13.33 | 13.3x |
| B | 3 | 1 | 8.0 | 8.0x |
| B | 5 | 1 | 8.0 | 8.0x |

No feasible point was found beyond the stated caps in either geometry —
consistent with (does not refute) the bound. **The "roughly factor of
two" overshoot reported for Sec 6.2 does NOT persist here.** The
observed slack is 4.4x-13.3x, substantially looser than the single-stop
geometry's ~2x. Two candidate explanations, not distinguished by this
run: either the bound is genuinely less tight for multi-component
geometries, or (more likely, given Sec 7's explicit warning) the true
feasibility boundary extends well past d=3/d=1 and the 250-300-start
budget simply failed to find those higher-d feasible points — the
sparse 6/56 feasibility rate throughout this run makes the latter
explanation hard to rule out. Reported as "cap not tight, by a wide and
geometry/n-dependent margin," not as a refutation of the bound itself.

## 6. P4 — tolerance structure

$\epsilon_0(N)=[1+\beta\sinh^2(N\mu_0)/\sinh^2\mu_0]^{-1}$ using $\beta=Q(\text{active point})-1$
at each active (C) point (max over active points), tested against
$\max_{I_0}T_N$ computed directly, for the best design in each geometry:

**Geometry A, n=5, d=1 (delta=4.68e-5)** — active stop component (I0[0]):

| N | max_TN direct | eps0 formula | agreement |
|---|---|---|---|
| 1 | 5.361783e-01 | 5.361783e-01 | 10 s.f. |
| 2 | 2.068677e-01 | 2.068677e-01 | 10 s.f. |
| 5 | 2.096529e-02 | 2.096529e-02 | 10 s.f. |
| 9 | 1.497200e-03 | 1.497200e-03 | 10 s.f. |
| 20 | 1.221669e-06 | 1.221669e-06 | 10 s.f. |

**Geometry B, n=3, d=1 (delta=1.124)** — its single stop component:

| N | max_TN direct | eps0 formula | agreement |
|---|---|---|---|
| 1 | 2.397851e-01 | 2.397851e-01 | 10 s.f. |
| 2 | 6.643821e-02 | 6.643821e-02 | 10 s.f. |
| 5 | 5.808985e-03 | 5.808985e-03 | 10 s.f. |
| 9 | 4.089599e-04 | 4.089599e-04 | 10 s.f. |
| 20 | 3.333354e-07 | 3.333354e-07 | 10 s.f. |

**P4 is confirmed, cleanly, on every active component tested** — the
generalised max-over-active-points formula reproduces the directly
computed $\max_{I_0}T_N$ to at least 10 significant figures at every
$N\in\{1,2,5,9,20\}$ checked, for both geometries. One informative
aside: Geometry A's *non-active* I0 component (I0[1], the one with 0
active (C) points at this optimum) does **not** follow the formula at
all (there is no active point to define a beta from) and instead decays
far faster — at N=20, max T_N there is ~2.6e-44 versus the active
component's 1.2e-6 — reflecting that it sits much deeper in the
stopband ($\kappa_{\min}=6.28$ there versus $\cosh\mu_0=1.05$ at the
binding component). The formula's domain of validity is exactly and
only the active components, as the prediction states.

## 7. eps_0/eps_1 table, best design per geometry, N=1,2,5,9

(eps_1(N) here is taken, by natural analogy with eps_0(N)=max_I0(T_N),
as 1-min_I1(T_N) -- the worst in-band transmission loss at that N. This
quantity is not redefined in this spec; the interpretation is disclosed
here since it was not given verbatim.)

**Geometry A best (n=5,d=1,sigma=(+1,-1), delta=4.68e-5):**

| N | eps_0 | eps_1 |
|---|---|---|
| 1 | 5.361783e-01 | 4.678941e-05 |
| 2 | 2.068677e-01 | 1.738977e-04 |
| 5 | 2.096529e-02 | 6.283174e-04 |
| 9 | 1.497200e-03 | 2.869454e-04 |

**Geometry B best (n=3,d=1,sigma=(-1), delta=1.124):**

| N | eps_0 | eps_1 |
|---|---|---|
| 1 | 2.397851e-01 | 5.291217e-01 |
| 2 | 6.643821e-02 | 5.919156e-01 |
| 5 | 5.808985e-03 | 6.204916e-01 |
| 9 | 4.089599e-04 | 5.591667e-01 |

Geometry A's eps_1 is small throughout (consistent with its small
delta); Geometry B's eps_1 sits around 0.53-0.62 at every N tested,
directly reflecting that its only feasible design has delta of order 1
— B's best available design in this search is simply a poor filter by
the pass-band criterion, not merely by the stop-band one.

## 8. Search-budget adequacy note

Every one of the 6 feasible cases had at least 20 converged starts
(range: 20 to 255 out of 250-300), so none is a single-lucky-start
fragile result in the sense flagged in the previous (single-geometry)
experiment's summary. However, **50 of the 56 cells returned zero
feasible starts out of a 250-302-start budget** — this is the dominant
pattern of this run, not the exception, and Sec 7's instruction to
report infeasibility as "none found within budget" applies to the large
majority of the enumerated grid, not to isolated cells. Whether the true
feasible region for d>=2/3 in these geometries is genuinely this sparse,
or the antisymmetric/random/continuation start families are simply
poorly matched to whatever the true feasible basins there look like, is
not resolved by this run.

## 9. Traps (Sec 7)

- theta=pi included in both geometries' grids (endpoint of a component
  in each); no near-zero-gradient issue was misread as solver failure —
  SLSQP's own res.success flag, not a gradient-magnitude heuristic, was
  the sole convergence criterion used.
- Geometry B's $A_{-1}=0$ prediction at even L: checked at both B
  optima (both are even-L: L=4 and L=6). B n=3 d=1: $A_{-1}$=-6.48e-09.
  B n=5 d=1: $A_{-1}$=3.72e-07. **Both consistent with exactly zero** to
  well within solver/grid tolerance — confirms the theory.
- Sign enumeration: sigma was swept independently at every d (never
  carried across d); Geometry A's sigma_2 additionally had to be
  re-derived at each d rather than reused verbatim in continuation (see
  config_multi.json's "sigma_free" note) since it flips between d and
  d-2.
- Infeasibility reported throughout as "none found within budget" (see
  Sec 8), never as a proof.

## 10. Anything Sec 6.2's structure does not explain

Two things: (i) the genuinely asymmetric (neither-symmetric-nor-
antisymmetric) optima at A n=3 d=1 sigma=(-1,+1) and B n=3 d=1 —
Sec 6.2's structure, built entirely around the antisymmetric ansatz,
has nothing to say about why these particular sign/geometry
combinations land off both symmetry axes rather than snapping to one;
(ii) the extra active point at A n=5 d=1, traced to the I1 interval
endpoint theta=pi/4 joining the active set — Sec 6.2's alternation count
implicitly assumes the equioscillation structure is entirely interior to
I1, and this case shows that assumption failing even in a case (small
delta, well-converged) that is otherwise a clean, high-confidence
result.
