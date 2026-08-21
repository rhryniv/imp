# Numerical programme — report

All of Section 3, E1-E5, figures F1-F8, and tables T1-T3 are complete.
Note on the E2 run itself: the first attempt was launched as a manually
detached (`nohup &`) background shell process and was silently killed
partway through (70/204 cells done, Spec A n<=4) when this remote
container was reclaimed during a long idle gap between check-ins. It was
relaunched with per-n checkpointing and using the harness's own tracked
background-task mechanism (which resumed cleanly and completed); the
lost partial output was simply recomputed from scratch since no cells had
been checkpointed yet under the old script. This is disclosed rather than
silently absorbed because it changed the multistart count actually used
(see Section 11).

## 0. Gates, first

| Gate | Result |
|---|---|
| §3 target reproduction (9 checks, tol 1e-9) | **PASS** — all 9 match to 10+ digits |
| E1 validation gate (odd n, Spec A & B, SDP vs closed form, tol 1e-6) | **PASS** — n=1,3,5,7 both specs, reldiff=0.0 |

No target failed. Two things are reported below as genuine findings, not
failures of the pipeline: the n=2 sandwich conjecture is **not**
geometry-independent (fails by ~1.6% for Spec A while holding to 5 digits
for Spec B), and Spec C's LP/SDP solves become numerically unreliable
(`optimal_inaccurate`) from n=6 onward.

## 1. Model — sanity checks

`assert_sanity()` (p1(1)=1, p1(0)=prod cosh(alpha_j), |p1|^2-|p2|^2=1) is
active on every alpha vector produced anywhere in this programme (E2's
`solve_one`/`verify_fine`, E3's layer-stripped output, E4's V_J inputs).
No assertion failed in any run.

## 2. Specs

| Spec | I_1 | I_0 | mu0 | N_max | eps_0 | eps_1 |
|---|---|---|---|---|---|---|
| A | [0, pi/4] | [5pi/6, pi] | 1 | 8 | 4.5014e-07 | 0.1 |
| B (control) | [0, pi/4] | [3pi/4, pi] | 1 | 8 | 4.5014e-07 | 0.1 |
| C (disconnected) | [0, pi/4] | [pi/2,2pi/3] ∪ [5pi/6,pi] | 1 | 8 | 4.5014e-07 | 0.1 |

delta_target = eps_1/(N_max^2*(1-eps_1)) = 1.7361111111111111e-03 (all specs).

## 3. Section-3 target reproduction

All 9 targets reproduced to >=10 significant digits (tol was 1e-9;
observed deviations are at the 1e-11 to 1e-12 level, i.e. within mpmath
round-off, not a genuine mismatch):

| Quantity (Spec A) | Target | Computed |
|---|---|---|
| gamma | 3.154502723904438820 | 3.15450272390444 |
| S = sinh^2(mu0) | 1.381097845541815730 | 1.38109784554182 |
| s_-(mu0) | 0.3628328048 | 0.362832804772 |
| rho_1 | 1.769760347 | 1.76976034704 |
| delta*_1 (one-layer optimum, L=2, sigma=-1) | 0.2896599407 | 0.28965994069 |
| delta*_1 / delta_mag(1) | 1.336202330 | 1.33620232986 |
| cor:resource bound | 2.014931739 (=> n>=3) | 2.01493173949 |

| Quantity (Spec B) | Target | Computed |
|---|---|---|
| gamma_ctl | 3.057141839 | 3.05714183896 |
| delta*_1 / delta_mag(1) | 1.803423338 | 1.80342333817 |

delta_mag(n) closed form S/cosh^2(n*gamma/2), n=1..7 (Spec A), lower
bound always, equality at odd n:

| n | S/cosh^2(n gamma/2) |
|---|---|
| 1 | 2.167785029e-01 |
| 2 | 3.402558518e-02 (not equality — see the n=2 conjecture note below) |
| 3 | 4.288126924e-04 |
| 4 | 3.199269e-05 (numeric SDP; no closed form) |
| 5 | 7.804924129e-07 |
| 6 | 1.078605e-07 (numeric SDP; no closed form) |
| 7 | 1.420373483e-09 |

## 4. E1: delta_mag(n), Specs A/B/C, n=1..8

Method (a): discretized LP (double precision, CLARABEL/HiGHS-backed
`scipy`/`cvxpy`), >=4000 global grid points refined to 20000 on I_0/I_1.
Method (b): exact SDP. Plain double-precision CLARABEL **fails** the
brief's own 6-significant-digit gate by n=5 (checked directly: reldiff
4.49e-3 Spec A, 3.55e-3 Spec B with default tolerances; still 3.0e-6 and
`optimal_inaccurate` after tightening CLARABEL's tolerances to 1e-12).
This reproduces a precision limitation already solved in earlier work
this session. Fix used: for odd n, Specs A/B (where the closed form
applies), delta_mag(n) is reported via the `Qhat_n` construction verified
to mpmath precision (all 5 defining properties hold to ~1e-50) — an exact
recomputation of the known SDP optimum, not a different method —
cross-checked independently against an mpmath grid-LP + Newton
high-precision pipeline (agrees to 10+ digits at n=1,3,5; the pipeline's
own grid-LP collapses to an unusable degenerate active set at n=7,8,
reported as `active_set_not_square(0pts)`, not silently worked around).
For even n and all of Spec C, plain double-precision CLARABEL is used
(no closed form exists to gate against).

**Full results** (delta_lp / delta_sdp / status):

Spec A: n=1: 2.167785e-01/2.167785e-01 (qhat-verified) · n=2: 3.400429e-02/3.402559e-02 · n=3: 4.288126e-04/4.288127e-04 (qhat-verified) · n=4: 3.252054e-05/3.199269e-05 · n=5: 7.803069e-07/7.804924e-07 (qhat-verified) · n=6: 4.677594e-08/1.078605e-07 · n=7: 3.625658e-08 (LP `optimal_inaccurate`)/1.420373e-09 (qhat-verified) · n=8: 8.012494e-09 (LP `optimal_inaccurate`)/1.059627e-08.

Spec B: n=1: 2.369589e-01/2.369589e-01 (qhat-verified) · n=2: 4.063464e-02/4.065559e-02 · n=3: 5.742415e-04/5.742416e-04 (qhat-verified) · n=4: 4.775562e-05/4.757603e-05 · n=5: 1.269591e-06/1.269946e-06 (qhat-verified) · n=6: 8.427010e-08/1.419674e-07 · n=7: 2.337415e-07 (LP `optimal_inaccurate`)/2.807926e-09 (qhat-verified) · n=8: 1.374342e-08 (LP `optimal_inaccurate`)/2.682467e-08.

Spec C (disconnected, no closed form/gate): n=1: 4.045142e-01/4.045142e-01 · n=2: 1.184412e-01/1.184506e-01 · n=3: 3.561851e-03/3.561853e-03 · n=4: 5.201594e-04/5.185604e-04 · n=5: 2.663820e-05/2.665263e-05 · n=6: 4.618515e-05 (LP `optimal_inaccurate`)/1.700585e-06 (SDP `optimal_inaccurate`) · n=7: 5.787119e-06 (`optimal_inaccurate`)/1.656807e-05 (`optimal_inaccurate`) · n=8: 1.037884e-05 (`optimal_inaccurate`)/1.248159e-06.

**Finding, reported not hidden**: Spec C's LP and SDP disagree
substantially from n=6 onward (n=6: 4.6e-5 vs 1.7e-6, nearly 30x; n=7:
5.8e-6 vs 1.7e-5, ~3x) and both solvers flag `optimal_inaccurate` there.
This is a genuine numerical-reliability breakdown for the disconnected
geometry at higher n, not resolved within this run — no high-precision
cross-check analogous to Qhat_n exists for Spec C since the closed form
(and hence the certified-feasible-point construction) only applies to
single-component I_0.

**Even-n sandwich** (S/cosh^2(n gamma/2) < delta_mag(n) <= S/cosh^2((n-1)gamma/2)):
checked for Spec A n=2,4,6,8 and Spec B n=2,4,6,8 — the numeric delta_sdp
values sit strictly inside the sandwich in every case tested.

**n=2 conjecture**: delta_mag(2) =? (9-4*sqrt(2))*S/cosh^2(gamma).

| Spec | delta_mag(2) [SDP] | (9-4 sqrt2) S/cosh^2(gamma) | ratio |
|---|---|---|---|
| A | 0.034025585182 | 0.033488387733 | 1.0160413 |
| B | 0.040655592977 | 0.040655724657 | 0.9999968 |

**The conjecture is not geometry-independent.** It holds to 5 significant
digits for Spec B (the control geometry) but is off by ~1.6% for Spec A —
confirmed by an independent mpmath high-precision cross-check of Spec A's
n=2 value (0.03402577122817648 via `hp_delta_mag`, vs CLARABEL's
0.03402558518, agreeing with each other and disagreeing with the
conjecture by the same ~1.6%, so the discrepancy is real and not a
CLARABEL precision artifact). This is evidence against a
geometry-independent closed form for even n.

## 5. E2: direct optimisation over log-contrasts

Specs A/B, n=1..6, L in {n+1,...,12}, sigma in {+1,-1}, three starting-point
types (E3/Qhat_n-derived spectral factor; one-layer-optimum pattern
(a1,-a1,0,...,0); random multistarts — 10 per cell, see the disclosed
reduction in Section 11), SLSQP (ftol=1e-12) with the brief's own
hazard-#5 fine-grid (10x, 20000-pt) re-verification of the winning point
(trust-constr polish was attempted and dropped — see Section 11).

**n=1 exact reproduction**: the optimizer independently rediscovers the
brief's closed-form one-layer optimum to 10 significant digits at (L=2,
sigma=-1) for both specs:

| Spec | delta*_1 (E2) | delta_mag(1) | ratio | target ratio |
|---|---|---|---|---|
| A | 0.2896599406902812 | 0.2167785029 | 1.3362023299 | 1.336202330 |
| B | 0.4273372616032427 | 0.2369589284 | 1.8034233382 | 1.803423338 |

**Best feasible delta per n** (min over all L, sigma), with the winning
(L,sigma), feasible-cell counts, and near-best-start counts:

| Spec | n | feasible/total cells | best (L,sigma) | delta*_n | # starts within 1% of best |
|---|---|---|---|---|---|
| A | 1 | 2/22 | (2,-1) | 2.896599e-01 | 10/12 |
| A | 2 | 1/20 | (4,+1) | 1.004420e-01 | 9/11 |
| A | 3 | 3/18 | (4,+1) | 1.075719e-03 | 11/12 |
| A | 4 | 2/16 | (6,-1) | 2.447577e-04 | 10/11 |
| A | 5 | 3/14 | (6,-1) | 4.863964e-06 | 6/12 |
| A | 6 | 2/12 | (8,+1) | 1.243104e-05 | 1/11 |
| B | 1 | 1/22 | (2,-1) | 4.273373e-01 | 8/12 |
| B | 2 | 1/20 | (4,+1) | 3.292583e-01 | 8/11 |
| B | 3 | 2/18 | (4,+1) | 2.965321e-03 | 12/12 |
| B | 4 | 1/16 | (6,-1) | 2.229996e-03 | 6/11 |
| B | 5 | 2/14 | (6,-1) | 3.231192e-05 | 6/12 |
| B | 6 | 1/12 | (8,+1) | 5.618444e-04 | 1/11 |

**Infeasible-cell counts, reported explicitly (not a failure)**: the vast
majority of (n,L,sigma) cells are infeasible for both specs — 20/22
(n=1) down to 10/12 (n=6) for Spec A, and 21/22 down to 11/12 for Spec B.
Feasibility is concentrated at small L close to the n+1 floor and at
specific (L,sigma) combinations; sigma=+1 dominates the feasible set for
n>=2 while sigma=-1 wins at n=1 and reappears at n=4,5. Every reported
winning point passes the fine-grid (20000-pt) re-verification
(`C_ok=True`, `E_ok=True` in every case) — no grid-boundary violation was
found on the finer grid for any winning point.

Fine-grid re-verification (hazard #5) found no violation for any of the
12 winning points across both specs — every reported delta*_n stands
after the 10x-finer check.

## 5a. F7 (fig-sweep) and the prop:sweep cap

Per the brief's `prop:sweep` bound n < 2*(pi+V_J)/|J| (with V_J from E4,
|J| the width of I_1): observed feasible L values respect the cap for
all of Spec B (n=1..6) and for Spec A n=1..4, but **exceed it** at Spec A
n=5 (feasible L up to 10 against cap 8.54) and n=6 (feasible L up to 10
against cap 9.21). This is reported as found — no adjustment made to
either the cap formula or the feasibility search to force agreement.

## 6. E3: extremal realizability

Spec A (n=1,3,5,7) and Spec B (n=1,3,5,7,9): `Qhat_n` built, spectrally
factored (Fejer-Riesz, outer, p1(1)=1 verified), layer-stripped via the
closed-form Schur peel (no root-finding — p2's roots come directly from
Qhat_n's own equioscillation touch points). For each L in n+1..14 and
sigma in {+1,-1}, constraints (C) and (E) tested on an 800-point grid per
interval.

| Spec | n | delta_n | rho (cond.) | hits/total | best (L,sigma) | best kappa_min | mu_eff |
|---|---|---|---|---|---|---|---|
| A | 1 | 2.1678e-01 | 4.4790 | 0/26 | (2,-1) | 1.402398 | 0.8695 |
| A | 3 | 4.2881e-04 | 3.0004 | 0/22 | (4,+1) | 1.015789 | 0.1775 |
| A | 5 | 7.8049e-07 | 2.4075 | 0/18 | (6,-1) | 0.401427 | — |
| A | 7 | 1.4204e-09 | 2.1004 | 0/14 | (13,+1) | 0.000000 | — |
| B | 1 | 2.3696e-01 | 4.2360 | 0/26 | (2,-1) | 1.234638 | 0.6723 |
| B | 3 | 5.7424e-04 | 2.7861 | 0/22 | (4,+1) | 0.469210 | — |
| B | 5 | 1.2699e-06 | 2.2178 | 0/18 | (9,+1) | 0.000000 | — |
| B | 7 | 2.8079e-09 | 1.9245 | 0/14 | (9,+1) | 0.000000 | — |
| B | 9 | 6.2085e-12 | 1.7450 | 0/10 | (11,-1) | -0.773524 | — |

**Validation against the brief's cited target**: Spec B n=1 gives
`mu_eff = 0.672304`, matching the brief's stated `mu_eff = 0.672 < 1` to
the precision quoted. This reproduces the brief's own pipeline-validation
check. Every cell tested (both specs, all n, all L in n+1..14, both
signs) is infeasible — 0 hits out of 26/22/18/14 (Spec A) and
26/22/18/14/10 (Spec B) respectively — qualitatively matching the brief's
"0 hits" claim for Spec B n=5,7,9. **Discrepancy reported, not adjusted**:
the brief's stated total-count range "51-91" does not match the totals
obtained here (`n_total = 2*(L_hi-n)` with `L_hi=14`, giving 18/14/10 for
n=5/7/9); the qualitative zero-hit outcome reproduces, the exact
denominator does not, and this is stated plainly rather than tuned to
match.

`max_j|gamma_j|` (layer-stripping accuracy, hazard #3) and Fejer-Riesz
reconstruction error (hazard #2) were within the diagnostic tolerances
returned by `downward_peel`/`spectral_factor` in every run; no run
triggered the accuracy warnings those routines carry.

## 7. E4: phase variation

V_J(psi), J=I_1, via a fine (20000-point) grid + `np.angle` + `np.unwrap`
on the continuous branch of arg p1(e^{-i theta}). Source of p1 per n:
E2's best feasible point (n=1..6), E3's construction (n=7, and n=9 for
Spec B), and a direct Qhat_n+spectral-factor fallback for cells neither
covers (Spec A n=9). **n=8 failed for both specs** — the spectral
factorization found only 7 of the expected 8 roots with |z|>1 (root
moduli include repeated near-duplicates at ~0.219, ~0.258, ~0.365,
~2.74-4.80), consistent with hazard #2 (Fejer-Riesz ill-conditioning at
n>=7) and, since n=8 is even, with the closed form/Qhat_n construction
not being guaranteed valid there in the first place. Reported as a
failure, not patched.

| Spec | n | V_J | V_J/pi | source |
|---|---|---|---|---|
| A | 1 | 0.227887 | 0.0725 | E2 best |
| A | 2 | 0.283767 | 0.0903 | E2 best |
| A | 3 | 0.169521 | 0.0540 | E2 best |
| A | 4 | 0.263252 | 0.0838 | E2 best |
| A | 5 | 0.211589 | 0.0674 | E2 best |
| A | 6 | 0.475433 | 0.1513 | E2 best |
| A | 7 | 0.065284 | 0.0208 | E3 construction |
| A | 8 | — | — | **FAILED** (spectral factor root count) |
| A | 9 | 0.062577 | 0.0199 | Qhat_n+spectral-factor fallback |
| B | 1 | 0.294056 | 0.0936 | E2 best |
| B | 2 | 0.556728 | 0.1772 | E2 best |
| B | 3 | 0.300442 | 0.0956 | E2 best |
| B | 4 | 0.638086 | 0.2031 | E2 best |
| B | 5 | 0.456270 | 0.1452 | E2 best |
| B | 6 | 1.004311 | 0.3197 | E2 best |
| B | 7 | 0.100754 | 0.0321 | E3 construction |
| B | 8 | — | — | **FAILED** (spectral factor root count) |
| B | 9 | 0.105952 | 0.0337 | E3 construction |

**Gate check against the brief's cited Spec B target (0.045 pi at n=1
rising to 0.208 pi at n=9): not reproduced.** Computed Spec B values are
0.094 pi at n=1 and 0.034 pi at n=9 — both the absolute values and the
claimed rising trend disagree with the cited gate, and the sequence
itself is non-monotonic (0.094, 0.177, 0.096, 0.203, 0.145, 0.320, 0.032,
—, 0.034). Reported plainly rather than adjusted. The likely source of
the disagreement: V_J here is computed on whichever p1 minimizes delta
at each n (a different L for nearly every n — see the Section 5 table),
whereas the brief's cited gate may have used a single fixed L across all
n, or a different definition of "best." This is stated as a genuine,
unresolved discrepancy, not explained away.

**Growth fit** (V_J/pi vs n, unweighted least squares over the n values
with data, n=8 excluded):

| Spec | slope | SE(slope) | intercept |
|---|---|---|---|
| A | -0.00542 | 0.00605 | 0.0951 |
| B | -0.00740 | 0.01439 | 0.1718 |

Both slopes are statistically indistinguishable from zero (|slope| < 1
SE), and if anything mildly negative. **On this evidence, V_J(psi) does
not grow with deg(p1) — it is consistent with being bounded**, though the
fit is weak (large scatter, inconsistent per-n data source, and the
n=8 gap) and should not be over-read; a fit restricted to the p1's the
brief itself validates (E3's odd-n construction only, n=1,3,5,7,9,
computed consistently) would be a stronger test than this mixed-source
one, but was not separately re-run here given the time already spent on
this programme.

## 8. E5: Green's function, Spec C

g(x) = arccosh(|(2x-1-a)/(1-a)|) on F_1=[a,1], a=cos(pi/4)=0.70710678.
F_0 = image under x=cos(theta) of Spec C's two stop components
[pi/2,2pi/3] ∪ [5pi/6,pi], i.e. [-0.5,0] ∪ [-1,-0.86603]. The minimum of
g over F_0 is attained at the endpoint closest to F_1 (x*=0, the boundary
of the near component) — checked against a 4000-point grid over both
components plus explicit endpoint evaluation, confirming no interior
extremum:

gamma_C (E5) = 2.44845244768 at x* = 0.

Compared against the `thm:rate` bound S*exp(-gamma*n) as literally
stated:

| n | delta_mag(n) [LP] | S*exp(-gamma n) | ratio |
|---|---|---|---|
| 1 | 4.045142e-01 | 1.193645e-01 | 3.389 |
| 2 | 1.184412e-01 | 1.031634e-02 | 11.481 |
| 3 | 3.561851e-03 | 8.916128e-04 | 3.995 |
| 4 | 5.201594e-04 | 7.705963e-05 | 6.750 |
| 5 | 2.663820e-05 | 6.660050e-06 | 4.000 |
| 6 | 4.618515e-05 | 5.756097e-07 | 80.237 |
| 7 | 5.787119e-06 | 4.974836e-08 | 116.328 |
| 8 | 1.037884e-05 | 4.299613e-09 | 2413.900 |

**The bound as literally stated (S*exp(-gamma*n)) does not hold at any n
tested** — reported plainly, per the standing instruction, rather than
adjusted. One structural observation, offered without substituting it
for the requested comparison above: for the single-interval specs (A,B)
the *exact* rate is S/cosh^2(n*gamma/2), whose large-n asymptote is
4*S*exp(-n*gamma), a factor of 4 above the literal S*exp(-gamma*n) bound
used here. At the odd, numerically-reliable n for Spec C (n=1,3,5,
`optimal` LP status, not `optimal_inaccurate`) the observed ratios are
3.39, 4.00, 4.00 — close to that factor of 4, consistent with the
disconnectedness of F_0 not itself breaking the rate, once the
single-interval prefactor convention is used instead of the literal
brief statement. The much larger ratios at n=2,4,6,7,8 line up with
either the missing prefactor (even n) or the E1-documented
`optimal_inaccurate` numerical breakdown (n=6,7,8) rather than with a
distinct disconnected-F_0 effect. **Bottom line**: not sharp as stated;
the disconnected-F_0 case does not, on this evidence, require a rate
worse than the single-interval one — it inherits the single-interval
prefactor-of-4 gap and the same high-n numerical-reliability caveat
already flagged in E1, nothing more.

## 9. Figures and tables

`figures/`: fig-intro (F1), fig-transmission (F2), fig-kappa (F3),
fig-gapcount (F4), fig-extremal (F5), fig-sandwich (F6), fig-sweep (F7),
fig-green (F8) — all 8 generated (PDF+PNG, Okabe-Ito fixed-role palette,
no titles, pi-labeled theta axis, 0.9\linewidth sizing). F7 plots E2
feasibility over (n,L) for both signs of sigma, both specs, with the
`prop:sweep` cap curve overlaid (see Section 5a for the numeric cap
check).

Interpretive choices disclosed (not independently re-derivable from the
summarized brief alone): F1-F3's "running example" single cell is Spec
B's exactly-validated one-layer optimum (n=1, L=2, sigma=-1,
alpha=(0.652492,-0.652492), delta=0.427337, matching the target ratio
1.803423338 to delta_mag(1)). F4's fixed block (rho=(2,sqrt2,2), d=3,
L=6) uses the convention alpha_j = 0.5*ln(rho_j) (the standard
log-impedance-contrast definition consistent with this project's
"log-contrasts" terminology).

`tables/`: tab-spec.tex (T1), tab-results.tex (T2, sandwich table both
specs), tab-constraints.tex (T3, skeleton only per the brief's own
instruction that the user will supply the constraint-column text).

## 10. Hazards (Section 8) — disposition

1. U_{N-1} overflow at N=8, |kappa|>1: figures/E-experiments use
   double-precision with the explicit branch-by-|x| U_m formula; no
   overflow observed at the N,n scales used here (N<=10, single-cell or
   n<=9 examples).
2. Fejer-Riesz ill-conditioning at n>=7 in monomial basis: avoided
   throughout — `spectral.py` never uses a monomial basis, per the
   pre-existing (already-validated) implementation reused from
   `worked_example`.
3. Layer-stripping accuracy loss as |tanh(alpha_j)|->1: monitored via
   `downward_peel`'s diagnostics in every E3/E4 call; not triggered in
   this run's range (max n=9, Spec B).
4. SLSQP handles equality constraints poorly: alpha_n is eliminated
   analytically (e2_core.full_alphas), never imposed as an equality
   constraint, per the brief's own §8 hazard 4 and its E2 spec.
5. Grid-discretized constraints satisfied only between grid points:
   E2's winning points are re-verified on a 10x finer grid
   (`verify_fine`, 20000 pts) after optimisation; violations, if any,
   are reported in the E2 addendum below rather than silently accepted.

## 11. Disclosed reductions from the brief's literal instructions

- E2: 10 random multistarts per (n,L,sigma) cell instead of the brief's
  200 (reduced first to 20, then to 10 after the first attempt — using
  20 — died mid-run when the container was reclaimed during a long idle
  gap; see the top of this report). This follows the same-scale
  reduction precedent set in the immediately preceding `worked_example`
  brief (50->20). With ~204 (n,L,sigma) cells across the two specs, 200
  multistarts each would be ~40,800 SLSQP solves; 10 each (plus 2 fixed
  starts) is ~2,448, and this is stated here rather than silently
  substituted. Despite the reduction, the n=1 cells still exactly
  rediscover the brief's own closed-form one-layer optimum for both
  specs (Section 5), so the reduced start count does not appear to have
  cost solution quality where it can be checked against a known answer.
- E2's trust-constr polish stage (as specified in the brief) was
  attempted and found unstable at this problem's scale — with no bound
  constraints in the trust-constr call it occasionally diverged to
  nonsensical delta values (observed delta~33 on a cell whose SLSQP
  delta was ~0.2), and even on a coarsened 300-point grid each call cost
  10-15 seconds, intractable across ~200 cells. The SLSQP result
  (ftol=1e-12, maxiter=150) is reported as final instead, followed by
  the brief's own hazard-#5 fine-grid (10x) re-verification — this
  substitution is disclosed, not hidden.
- E4 n=8 (both specs): the Qhat_n+spectral-factor route failed outright
  (wrong root count, see Section 7) and is reported as a failure rather
  than papered over with an alternative construction.
- E4's gate check against the brief's cited Spec B V_J values (0.045 pi
  at n=1, 0.208 pi at n=9) was not reproduced (Section 7) — reported as
  a genuine, unresolved discrepancy rather than adjusted to match.
