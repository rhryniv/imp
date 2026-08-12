# Run log: spec Sec. 8 decisions

**Superseded in part by the manuscript revision** (see driver.py's own
module docstring for the full account): the pass-band admissibility test
this log's items 4-6 describe as using `s_0` is no longer how `admissible`
is computed (that gate is now `max_kappa_B<=1` + `delta<=delta_target`,
`s_0` kept only as a diagnostic), `mu0` is now derived from a top-level
`N_max` rather than supplied directly, and the `N_required` formula
changed from `arccosh(eps0**-0.5)/mu_min` to
`ceil(log(4/eps0)/(2*mu_min))`. This log is left as-is below as an
accurate record of the decisions made AT THE TIME under the original
formulation; it is not rewritten to match the revision.

codespec6.4.md Sec. 8 ("Where the code has authority") lists six points
where the spec asks for "a recommendation with evidence; the manuscript
follows," rather than a literal prescription. This log records the
decision actually made at each point, the evidence behind it, and where
it lives in the code. Deliverable 4 of Sec. 9.

Two of the six (root-finding basis; on-circle tolerance) were decided
together, early in Stage 2, and are reported together below. The other
four were each decided in the course of building Stage 2/3, several
after a deviation-forcing problem was found empirically and reported to
the user before being fixed (per the spec's own Authority rule) --
those are cross-referenced to `direct.py`'s own module docstring, which
carries the full evidence, rather than duplicated verbatim here.

---

## 1. Root-finding basis, and 3. on-circle tolerance for retained roots

**Decision:** Chebyshev/colleague-matrix root-finding in `x = cos(theta)`
(`certify.py`, `_extremize_chebyshev`), not a monomial companion matrix
in `z = exp(i*theta)`. This was the spec's own suggested alternative
(Sec. 8's second bullet names it explicitly), and it was adopted outright
rather than implemented as an optional variant to compare against.

**Reasoning:**
- `Q(theta)` and `kappa_B(theta)` are already Chebyshev series in
  `x = cos(theta)` (Stage 1's own coefficient conventions -- `Q`'s
  coefficients are the doubled autocorrelation, `kappa_B`'s are the
  block's own coefficient vector `a`, since `T_m(x) = cos(m*theta)`), so
  no conversion step is needed to get into this basis; it is the native
  one.
- Real-valued colleague-matrix eigenvalues (`numpy.polynomial.chebyshev.
  Chebyshev.deriv().roots()`) are exactly the critical points of a real
  trig polynomial restricted to a real sub-interval -- there is no
  "spurious root" category analogous to the z-domain approach's
  off-circle numerical artifacts, only ordinary complex-eigenvalue
  rounding noise, filtered by a single, unrelated tolerance
  (`root_tol=1e-9` in `_extremize_chebyshev`, keeping eigenvalues with
  `|imag| < root_tol`, itself unrelated to the old on-circle question).
- Consequence for bullet 3: the z-domain approach's own "on-circle
  tolerance for retained roots" question is not answered by tuning a
  parameter -- it is *avoided entirely* by this basis choice. The
  z-domain implementation this project used before this rewrite
  (`git show 7af08d0`, now superseded) used `circle_tol = 1e-6` to filter
  companion-matrix roots that were supposed to lie on `|z|=1` but didn't,
  exactly due to floating-point noise in a step this approach removes
  structurally. No such tolerance exists in the current code.
- Degree halves for free, as the spec's own Sec. 8 text anticipates: a
  degree-`n` trig polynomial's critical points are found via a
  real-coefficient Chebyshev colleague matrix of size linear in `n`,
  versus a z-domain companion matrix on the cleared-of-negative-powers
  Laurent polynomial, whose degree (and hence matrix size) is twice as
  large before any on-circle filtering is even applied.

**Code:** `scattering/certify.py`, module docstring and
`_extremize_chebyshev`.

---

## 2. Grid size `M` and node placement

**Decision:** Uniform grids in `theta`, `M = 200` points per interval
component by default (`n_grid_B`, `n_grid_C` in
`direct.py:design_direct_literal`, threaded through
`driver.py:run_degree_stage3`) -- not refined near the endpoints of
`I1`.

**Reasoning:** The grid is used only to pose the SQP's finitely-many
pointwise constraints (Phase 1's stop-band depth constraints, Phase 2's
pass-band flatness and stop-band depth constraints); it plays no role in
the *reported* numbers, since every reported `delta`, `kappa_min`,
`mu_min`, `s_0` in a `DegreeRecord` comes from `certify_exact`'s own
exact, grid-independent rootfinding, not from evaluating the SQP's grid.
A uniform grid was kept rather than refining near `I1`'s endpoints
because (a) `certify_exact`'s own extremization already finds the true
worst point exactly regardless of where the SQP's grid happened to
sample, so a coarse SQP grid cannot silently produce a wrong *reported*
number, only a design whose *achieved* margin at the true worst point is
slightly worse than the SQP believed; and (b) `grid_vs_exact` (spec
Sec. 6) is emitted precisely so this gap is visible per degree rather
than assumed away -- see the `grid_vs_exact` column in any
`degree_scan_stage3` output. No case encountered while building Stage 2/3
needed endpoint refinement to converge; if a future instance shows a
persistently large `grid_vs_exact` discrepancy, that is the signal to
revisit this default, not a fixed grid density chosen in advance.

**Code:** `scattering/direct.py:design_direct_literal` (`n_grid_B=200,
n_grid_C=200`); `scattering/driver.py:run_degree_stage3` (same
defaults, passed through).

---

## 4. SQP implementation and tolerances

**Decision:** `scipy.optimize.minimize(method="SLSQP")` throughout
(Phase 1 and Phase 2), with exact analytic gradients
(`forward.forward_with_grad`) supplied to every objective and constraint
rather than SLSQP's own finite-difference Jacobian estimate; the
optimization variable is `gamma_free = tanh(alpha_free)` (bounded to
`(-gamma_bound, gamma_bound)`, `gamma_bound = 0.9995`), with the
`d(alpha_free)/d(gamma_free) = 1/(1-gamma_free^2)` chain-rule factor
applied explicitly to every constraint Jacobian built on `gamma_free`.

**Concrete tolerances:**
- `maxiter=400`, `ftol=1e-14` (SLSQP's own options) throughout.
- Phase 1 (`phase1_maximize_depth`): ramped-target feasibility search
  (see item 6 below for why), `n_steps=20` ramp stages, `max_bisections=4`
  step-halvings on a stage that fails to verify, quadratic regularizer
  weight `reg=1e-4` (small relative to the constraint, only to keep the
  feasibility solve well-posed away from the target boundary), verified
  post-hoc at tolerance `1e-4` on the depth constraint.
- Phase 2 (`phase2_flatten`): verified post-hoc at tolerance `1e-6` on
  both the flatness and depth constraints, plus an outright rejection of
  any converged point with `t > 100` (a "locally stationary but
  nonsensical" filter -- SLSQP occasionally converges a badly-scaled seed
  to a point that technically satisfies the pointwise grid inequalities
  with an enormous `t`; a genuinely useful design's `delta` is never
  remotely close to 100).
- The gradient chain-rule fix itself was found, not assumed: the
  superseded `sdp_design.py` implementation of this same SQP omitted the
  `1/(1-gamma_free^2)` factor, treating the tanh-space gradient as if it
  were the alpha-space gradient directly -- confirmed to be imprecise
  (not silently wrong on any previously-reported result, but genuinely
  misleading to SLSQP's local QP step near the box boundary, where the
  true factor is large) before being corrected in `direct.py`. See
  `direct.py`'s own module docstring for the full account.
- The exact analytic gradients themselves were validated against a
  4th-order (5-point) finite-difference stencil at the spec's own
  required precision bar, relative error `< 1e-7`, across `n = 1, 2, 4,
  7, 12` (`tests/test_stage2_gradients.py`) -- a plain 2-point central
  difference was checked and confirmed *unable* to meet that bar on its
  own terms (its own truncation/rounding noise floor exceeds `1e-7`
  relative error on moderate-magnitude gradient components), which is
  why the test's own reference method is the 5-point stencil, not a
  cruder one.

**Code:** `scattering/direct.py` (`phase1_maximize_depth`,
`phase2_flatten`, `_chain_rule_scale`); `scattering/forward.py`
(`forward_with_grad`, `grad_free_vars`); `tests/test_stage2_gradients.py`.

---

## 5. Number and distribution of random starts

**Decision:**
- Phase 1 (`phase1_maximize_depth`, one call per sign pattern): up to 5
  perturbation scales tried in increasing order,
  `perturb_scales=(1e-3, 1e-2, 0.05, 0.1, 0.2)`, each drawn once as
  `gamma_free_0 ~ Normal(0, scale) in R^n`; stops early the first time a
  scale reaches the target depth, otherwise keeps the attempt reaching
  the greatest depth across all 5. Why several scales rather than one
  fixed value: `alpha=0` is a genuine gradient degeneracy (`Q-1` and
  `d(kappa)/d(alpha_j)` both vanish there, since every layer matrix is
  diagonal at `alpha=0`), so a single fixed perturbation scale is not
  reliably enough to escape it for every target depth `mu_0` and degree
  `n` -- confirmed empirically while building Phase 1, not assumed.
- Phase 2 (`build_seed_pool`, one pool per sign pattern per degree): the
  Phase-1 point itself; ONE zero-padded embedding *per smaller degree
  already in the scan* (`_padded_alpha_seeds`' first/trailing-zero
  variant only, out of its `max_variants=4`) -- not just the immediately
  preceding degree, every smaller degree `degree_scan_stage3` has
  already solved, per rule 5; `n_random=3` random draws at a small fixed
  scale (`0.02 * Normal(0,1)^n`, deliberately small -- these are meant to
  probe near the already-good Phase-1/padded points, not explore
  broadly); and, off by default, a fourth *investigative* member from the
  magnitude-SDP factorization (`magnitude_seed`), added only to answer
  the manuscript's own open question about that warm start (tracked via
  `start_origin`), not used as a routine seed.

  Originally all 4 of `_padded_alpha_seeds`' variants were kept per
  smaller degree; found empirically, while running the actual n=2..16
  Instance 1/2 scans, that this makes the pool (and hence the number of
  Phase 2 SLSQP solves per degree) grow ~4x faster than needed -- by
  n=16 that is ~14 prior degrees x 4 variants, per sign pattern for
  multi-band instances, and neither scan had finished after 48 minutes.
  Reported to the user with that evidence before cutting to 1 variant per
  smaller degree (kept: the trailing-zero placement) -- the informative
  part of a padded seed is which smaller degree it came from, not which
  of 3-4 structurally similar placements of the padding was used.
- No member of either pool is screened for constraint feasibility before
  being handed to SLSQP -- deliberate, not an oversight: the spec is
  explicit that zero-padded starts are typically infeasible by
  construction (`L` shifts from `2n` to `2(n+m)` under the padding), so a
  feasibility screen would discard exactly the seeds most likely to be
  useful.

**Code:** `scattering/direct.py` (`phase1_maximize_depth`'s
`perturb_scales`; `build_seed_pool`); `scattering/sdp_design.py`
(`_padded_alpha_seeds`).

---

## 6. Whether Phase 1 converges reliably enough for the cross-pattern comparison

**Decision:** Not as literally specified, and not without changing what
gets compared. Two deviations, both found empirically, both reported to
the user with concrete evidence before being implemented (per the
Authority rule), both approved:

1. **Phase 1 itself is a ramped-target feasibility search**, not the
   literal unconstrained epigraph maximization `max_{alpha,t} t`. The
   literal form has no interior optimum -- more contrast always helps
   `kappa_min` -- so it drives every free `alpha_j` to exactly the box
   boundary regardless of how loose the box is. Confirmed on the n=7
   multiband case: `kappa_min` reached `~2.2e9`, `alpha` components
   pinned at the box edge, and the resulting Phase 2 seed gave
   `delta=57.9` instead of the known-good `0.0277671370191698`. Without
   this fix, Phase 1 does not "converge reliably enough for cross-pattern
   comparison" in any useful sense: every pattern's `kappa_min` diverges
   to the same box-limited ceiling, destroying the comparison entirely
   rather than merely degrading it.
2. Even after that fix, **Phase 1's own largest-`kappa_min` ranking is
   not a reliable predictor of which sign pattern actually flattens
   well in Phase 2** -- confirmed on the same n=7 case: the
   largest-margin pattern `sigma=(-1,-1)` (`kappa_min=1.926`) gave
   `delta=57.9`, while the discarded `sigma=(-1,1)`
   (`kappa_min=1.00125`, barely clearing the target) gave the correct
   `delta=0.0277671...`. So Phase 2 is run on *every* Phase-1-feasible
   pattern, not just the single largest-margin one, and the best
   (minimum `delta`) result is kept. Cost is modest in the spec's own
   instances (`m_0` typically 1-3, so at most `2^{m_0} <= 8` Phase 2
   solves per degree).

Both are documented at length, with the full derivation and numbers, in
`direct.py`'s own module docstring -- not duplicated verbatim here to
avoid the two copies drifting apart; that docstring is the canonical
record.

**Code:** `scattering/direct.py`, module docstring,
`phase1_maximize_depth`, `phase1_all_sign_patterns`,
`design_direct_literal`.

---

## Related decisions outside the six-item list

Not asked for by Sec. 8's own enumeration, but made under the same
"report a recommendation with evidence" spirit, since the spec does not
pin these down either:

- **SDP solver: CLARABEL** (`cvxpy`'s default interior-point conic
  solver used throughout `dual_certify.py`/`sdp_design.py`), chosen
  because it is the solver against which the magnitude SDP's own dual
  sign convention was empirically pinned down (see
  `dual_certify.py`'s module docstring) -- a different solver is not
  guaranteed to share the same convention without re-deriving it.
- **Dual-feasibility PSD tolerance: `1e-6`**
  (`dual_certify.solve_magnitude_sdp_with_duals`'s `tol` parameter,
  applied to the minimum eigenvalue of every Hankel/localizing dual
  witness matrix). Chosen empirically: toy problems with a known
  closed-form optimum (`tests/test_stage3_dual_certify.py`) reproduced
  PSD witnesses with minimum eigenvalues in the `1e-8`-to-`1e-11` range
  from solver noise alone: `1e-6` sits comfortably above that noise
  floor without being loose enough to wave through a genuinely
  indefinite matrix.
