# Geometry C — two pass components — summary

## 1. Gate and probe validation

**Gate**: Sec 6.2 geometry, n=3,d=1,sigma=+1: delta=4.6722221733e-04
against target 4.6722e-4 — **PASS**.

**Probe recovery** (n=5,d=1,sigma=-1, cited delta~=3.29e-2): recovered
**delta=0.032871412424680946** — matches to the precision quoted.
**PASS.** No explicit alpha vector was given in the brief for this
"probe seed," so it was not injected as a dedicated start type; the
cell was solved with the standard random+antisym+continuation starts
only, and the recovered value is a genuine post-hoc validation, not a
seeded result (disclosed in config_C.json).

## 2. The optima found — sparser than either earlier geometry

**Only 2 of 15 cases feasible (13%)**, both at d=1:

| n | d | sigma | delta | n_converged/total | quality |
|---|---|---|---|---|---|
| 5 | 1 | -1 | 3.287141e-02 | 178/250 | robust |
| 7 | 1 | +1 | 1.722164e-02 | 1/251 | **fragile — see Sec 6** |

**n=3 is infeasible at every d tested (all 5 cells, 0/250 converged in
every one)** — a new finding this geometry did not anticipate (the
brief's open question was about n=7, not n=3). This is reported as
"none found within budget," not as proof; see Sec 8.

**n=7 resolves the brief's own flagged question** ("the earlier probe
found nothing at 40 starts while n=5 succeeded"): with 251 starts, this
run found exactly **1** feasible point at n=7,d=1 — consistent with a
40-start probe finding nothing (the true feasible basin here is small
enough that ~250 starts barely clears it). Given only 1/251 converged,
this result is flagged as low-confidence throughout this summary; it
resolves the question qualitatively (n=7 IS feasible, just barely) but
its exact numbers should not be trusted the way n=5's are.

### delta vs d, per n (— = infeasible; sigma forced, shown per row)

| n | d=1 | d=3 | d=5 | d=7 | d=9 |
|---|---|---|---|---|---|
| 3 (sigma: +1,-1,+1,-1,+1) | — | — | — | — | — |
| 5 (sigma: -1,+1,-1,+1,-1) | **3.287141e-02** | — | — | — | — |
| 7 (sigma: +1,-1,+1,-1,+1) | 1.722164e-02 (fragile) | — | — | — | — |

## 3. P1 — alternation count, p=(n+3)/2-a-e

| Case | n | predicted p | observed p (total) | a | e | match? |
|---|---|---|---|---|---|---|
| n5_d1 | 5 | (5+3)/2-1-0=3 | 1 | 1 | 0 | **refuted** |
| n7_d1 | 7 | (7+3)/2-0-1=4 | 1 | 0 | 1 | **refuted** (low confidence) |

**Both cases refute P1**, and unlike Geometry A's n=5 refutation (a
one-point overshoot from an endpoint effect), here the observed p is
*far* below prediction in both cases (1 vs 3, 1 vs 4). n5_d1 is the
robust, well-converged case, so this refutation carries real weight —
P1 does not survive the transition from varying $m_0$ (Geometry A,
where it held at n=3 and nearly held at n=5) to varying $m_1$ instead.

A genuinely new structural fact surfaced at n7_d1: **a=0** — the (C)
constraint is not active at all (kappa_min_I0=4.23, far above
cosh(mu0)=1.05); this optimum is instead pinned by (E) (e=1) and,
presumably, (B) alone. No earlier case in any of the three geometries
run so far had a=0. This alone would break the alternation-count formula
as stated, since it assumes an active-(C)-point contribution.

The theta~0 degeneracy the spec warned about (Sec 5 item 5) was checked
directly at n5_d1: the nearest near-touch to |kappa|=1 other than the
genuine active point sits at theta=0.00004 (immediately next to the
excluded theta=0 node), with residual -8.0e-09 — just outside the 1e-9
tolerance and correctly excluded. This is exactly the artifact the spec
described, confirmed by direct inspection rather than assumed.

## 4. P2 — symmetry

| Case | antisym. residual | sym. residual | classification |
|---|---|---|---|
| n5_d1 | 1.40e-09 | 0.946 | **antisymmetric** |
| n7_d1 | 3.29e-05 | 1.260 | **antisymmetric** (to the precision this fragile optimum was found at) |

Both feasible optima in Geometry C are antisymmetric — no broken-symmetry
case like Geometry A's turned up here (though with only 2 data points,
this is thin evidence either way).

## 5. P3 — least period

**Cannot be meaningfully tested.** At most one d was found feasible per
n (n=3: none; n=5: only d=1; n=7: only d=1), so there is no head-to-head
comparison available the way Geometry A's n=3 (d=1 vs d=3, 18x margin)
allowed. d=1 is the *only* feasible d found at n=5 and n=7, which is
consistent with "d=1 optimal" but is not evidence for it in the way a
genuine comparison would be — it may equally mean higher d is feasible
but unfound within budget (see Sec 8).

## 6. P4 — tolerance formula

**n5_d1** (a=1, beta=7.107190820663362):

| N | eps0 direct | eps0 formula | agreement |
|---|---|---|---|
| 1 | 1.233473e-01 | 1.233472879e-01 | 7 s.f. |
| 2 | 3.076934e-02 | 3.076933676e-02 | 7 s.f. |
| 5 | 2.599657e-03 | 2.599657209e-03 | 7 s.f. |
| 9 | 1.824716e-04 | 1.824715804e-04 | 7 s.f. |
| 20 | 1.486955e-07 | 1.486954598e-07 | 7 s.f. |

**Confirmed** to at least 7 significant figures at every N — consistent
with the 10-s.f. agreement seen in the two earlier geometries (this
one's agreement is nominally lower only because fewer digits were
printed in the source values, not because the fit is worse).

**n7_d1: not testable.** a=0 (no active (C) point at all — see Sec 3),
so there is no beta to build the formula from. eps0 direct still exists
and decays extremely fast (1.09e-2 at N=1 down to 1.07e-37 at N=20) —
far faster than n5_d1's — reflecting that (C) is deeply slack there, not
marginally active.

## 7. P5 — ripple distribution across the two pass components (new)

| Case | delta, component 1 [0,0.20pi] | delta, component 2 [0.40pi,0.55pi] | absolute gap | relative gap |
|---|---|---|---|---|
| n5_d1 | 0.03287136746732644 | 0.032871412424680946 | 4.5e-08 | 1.4e-06 |
| n7_d1 | 0.017221573532317347 | 0.017221638897607194 | 6.5e-08 | 3.8e-06 |

**In both cases, the two pass components come extremely close to the
same ripple level — within about one part in a million — but strictly,
only component 2 attains the true global delta** (component 2 is where
p_per_component registers 1 in both cases; component 1 registers 0,
falling short of delta by 4.5e-08 to 6.5e-08 in absolute terms, both
comfortably outside the 1e-9 active-point tolerance). So: **the
equioscillation structure nearly, but not exactly, spreads evenly across
both components** — it strongly favours doing so (to within 1e-6
relative) without the two components becoming truly degenerate at the
tolerance this experiment resolves. Whether tightening n_random/n_antisym
or running at higher precision would close that residual gap to exactly
zero, or whether it is a genuine (if very small) structural asymmetry,
is not resolved by this run.

## 8. Search-budget adequacy — the dominant finding of this run

**13/15 cells (87%) returned zero feasible starts within a 250-252-start
budget**, and even one of the two feasible cells (n7_d1) succeeded on
only 1 of 251 starts. This is a more severe version of the pattern
flagged in the previous (Geometry A/B) run, where 50/56 cells (89%)
were empty — comparable in degree, but here it also swallows an entire
n value (n=3) rather than being scattered across n and d. **Per Sec 8's
explicit instruction: this recurrence means the comparison across
geometries A, B, C is at real risk of measuring search difficulty rather
than design difficulty**, and that risk should be weighed against any
cross-geometry claim (e.g. "Geometry C has no feasible n=3 design") --
what this run actually shows is "no feasible n=3 design was found by
this start-generation strategy within this budget," which is a weaker
and different claim.

Cells with fewer than ~10 converged feasible starts: n7_d1 (1
converged) is the only *feasible* cell in this category. All 13
infeasible cells had 0 converged by definition.

## 9. eps table for the best design per n (N=1,2,5,9,20)

n=3 has no feasible design (see Sec 2/8) — omitted.

**n=5 best (d=1, delta=3.287e-2):**

| N | eps0 | eps1 |
|---|---|---|
| 1 | 1.233473e-01 | 3.182527e-02 |
| 2 | 3.076934e-02 | 1.002172e-01 |
| 5 | 2.599657e-03 | 1.498693e-01 |
| 9 | 1.824716e-04 | 8.571024e-02 |
| 20 | 1.486955e-07 | 1.721667e-01 |

**n=7 best (d=1, delta=1.722e-2, low-confidence — 1/251 converged):**

| N | eps0 | eps1 |
|---|---|---|
| 1 | 1.094443e-02 | 1.693008e-02 |
| 2 | 1.546129e-04 | 4.857602e-02 |
| 5 | 4.595751e-10 | 5.870444e-02 |
| 9 | 1.964811e-17 | 6.223195e-02 |
| 20 | 1.067757e-37 | 6.002254e-02 |

n=7's eps0 decays far faster than n=5's (consistent with a=0 there —
deep in the slack stopband, not marginally constrained), while its eps1
is non-monotonic in N for both n=5 and n=7 (rises then falls) — an
in-band transmission-loss pattern with no obvious simple form, reported
as observed.

## 10. Cross-check (Sec 8 trap)

Q via autocorrelation f_m vs direct |p1(e^{-i theta})|^2, spot-checked
every 500th fine-grid node at both optima: n5_d1 max diff 3.11e-15,
n7_d1 max diff 4.06e-14. Both far below the 1e-10 reporting threshold —
no conditioning issue.

## 11. Something the earlier geometries do not explain

Two things, both at n7_d1: (i) **a=0** — an optimum where the stop-band
constraint (C) is not active at all, which never occurred in Sec 6.2 or
either earlier multi-component geometry, and which by itself falsifies
the implicit assumption behind P1's formula (that exactly one
constraint-family beyond (B) is always active); (ii) the extreme
fragility itself (1 of 251 starts) at a point where n=5's analogous cell
was easy (178/251) — the qualitative jump in difficulty from n=5 to n=7
is much sharper here than anything seen varying $m_0$ (Geometry A) or
$m_1$ with $\pi\in I_1$ (the abandoned Geometry B), suggesting the
$m_1=2$, $\pi\in I_0$ combination specifically narrows the feasible
region very fast with n in a way neither earlier change did on its own.
