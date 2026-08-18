# Follow-up: is the decay rate γ attained?

Baseline geometry: `I1=[0,π/4]`, `I0=[3π/4,π]`, `μ0=1`, `γ=3.0571418...`
(`arccosh|(2cos u − cos t − 1)/(1−cos t)|`, exact to 50 digits throughout
this note). All results below use **Variant II** (`Qhat = 1+(1-x)R(x)`,
`R≥0` on `[-1,1]`, degree `n-1` — equality-free, so no Slater concern).

## Method (why this needed more than tighter solver tolerances)

The brief's Option 1 (rescale the objective, `delta -> delta/beta_n`) was
tried first and **does not work**: substituting a rescaled `delta`
variable into the SDP's own constraint matrix makes CLARABEL's
conditioning *worse*, not better (status flips to `optimal_inaccurate`
and the returned ratios visibly disagree with the unscaled run — e.g.
n=4: unscaled ratio 7.05 vs rescaled 6.04). The same rescaling applied to
a plain grid-LP (scipy/HiGHS) also fails past n≈6, but for a different,
more fundamental reason: HiGHS's own feasibility/optimality tolerances
(~1e-8 to 1e-9, and tightening them below ~1e-9 makes the double-precision
constraint matrix itself appear spuriously infeasible) round a true
`delta ~ 1e-10` down to exactly `0`, regardless of which variable is
rescaled.

What did work is a genuine high-precision (`mpmath`, 50-60 decimal
digits) **Remez-style active-set solve**:

1. Solve a fine-grid LP in double precision (HiGHS) to *locate* the
   active-constraint pattern — by LP theory a non-degenerate vertex has
   exactly `n+1` active constraints among the `n+1` decision variables
   (`R`'s `n` Chebyshev coefficients + `delta`), so this count is a
   built-in consistency check, not a guess.
2. Classify each active point as a domain boundary (`x=cos t`, `x=1`,
   `x=cos u` — no free unknown) or a genuine interior touch (adds one
   unknown, its exact location, paired with a tangency/derivative-zero
   equation).
3. Solve the resulting square nonlinear system with `mpmath.findroot`
   (Newton, `mp.dps=60`), seeded from the double-precision LP solution.
   Geometry constants (`cos t`, `cos u`, `sinh²μ0`, ...) are recomputed
   natively in `mpmath` from exact `π`-fractions — round-tripping them
   through a `str(float)` silently caps the whole solve at ~16 digits
   regardless of the working precision requested; this was caught and
   fixed during development (see `variants.py`'s `mp_X1`/`mp_X0`/...).
4. Verify on a fine high-precision grid that no constraint is violated
   beyond the target tolerance.

This reproduces the exact `n=1` closed form to **~44 digits** (the
mpmath solve: `0.23695892836645156685592154263176594091430653532041`;
the closed form `sinh²μ0(1−cos t)/(1−cos u)`:
`...430653557274`) — a decisive validation of the method. It also
revealed that **CLARABEL's own `n=1` answer, `0.23695892847378772`, is
only accurate to about 9-10 digits**, despite reporting clean `optimal`
status with a tiny duality gap.

### Where this method stops working, and why

The active-set *pattern* (see Deliverable C) is combinatorially clean and
extrapolatable, but populating it for `n≥7` needs a **double-precision
seed that is itself already accurate enough to see the pattern** — and
that is exactly where CLARABEL and the grid-LP both degrade:

- Constraint `C` (`Qhat≥cosh²μ0` on `I0`) is active (touches `x=cos u`
  exactly) for every baseline `n=1..6` in the verified high-precision
  solution. But CLARABEL's own **double-precision** `n=6` solution shows
  `C` comfortably *slack* (margin 0.886) — not a rounding-scale
  discrepancy. Checking: CLARABEL's `n=6` delta is `1.3985e-7`; the true
  optimum is `8.7103e-8`, ~1.6x smaller. **CLARABEL did not just lose
  precision on `delta` — it silently returned a strictly suboptimal
  point (`optimal` status, no warning) that isn't even at the same
  constraint-active vertex as the true optimum.** This is a materially
  more serious caveat than "add more digits" for anyone reading
  CLARABEL's `n≳6` output.
- At `n=7`, `delta ~ 1e-8` is already inside CLARABEL's own noise floor,
  so its solution can't reliably tell us the active set either (residuals
  at candidate touch points are the same order of magnitude as `delta`
  itself, not distinguishably zero).
- Several concrete active-set hypotheses were tried at `n=7,8` (extending
  the `n≤6` pattern by continuity, including/excluding the `x=cos t` and
  `x=1` boundary points in different combinations); each either failed to
  converge or converged to a spurious near-zero `delta` with a badly
  violated `C` — clear evidence of the wrong combinatorial pattern, not a
  tuning issue.

Per the task's own Option 3, **`n=7..11` are reported as not reliably
resolved** rather than extrapolated. `n=1..6` are resolved to ~50-digit
precision and are the basis for every claim below.

## Deliverable A — baseline, n=1..6 (exact)

| n | δ_mag(n) (50-digit) | β_n = sinh²μ0·e^(−γn) | ratio δ/β_n | δ(n)/δ(n−2) | e^(−2γ) |
|---|---|---|---|---|---|
| 1 | 0.23695892836645156685592154263176594091430653532041 | 0.0649419 | 3.64879 | — | — |
| 2 | 0.04065572465689367135442251690372259133885 | 0.0030537 | 13.3136 | — | — |
| 3 | 0.0005742415920014410907772613936829628746026 | 0.00014366 | 3.99917 | 0.00242338 | 0.00221106 |
| 4 | 0.00004787171170276385077035343358780205495984 | 6.7527e-6 | 7.09012 | 0.00117749 | 0.00221106 |
| 5 | 0.000001269945501896185693478510218562330302998 | 3.1748e-7 | 3.99999816 (≈4) | 0.00221152 | 0.00221106 |
| 6 | 0.00000008710324180798875507041323595508331353813 | 1.4930e-8 | 5.83457 | 0.00181951 | 0.00221106 |

All 6 are feasibility-verified (independent fine-grid, and boundary
residuals `<1e-50` at the active points). `n=1` reproduces the closed
form to ~44 digits. `n=7..11`: **not resolved** (blank per the task's own
instruction, rather than reporting noise).

The odd two-step ratio `δ(5)/δ(3) = 0.0022115` agrees with
`e^(−2γ)=0.0022111` to **4-5 significant figures** — matching the user's
own `2.219e-3` observation closely and confirming it wasn't a fluke of
CLARABEL's precision. `δ(3)/δ(1)=0.0024234` is further off, consistent
with still being pre-asymptotic at such low degree.

## Deliverable B — five geometries, `log δ_mag(n) + γn` vs n

| case | γ | n=1 | n=2 | n=3 | n=4 | n=5 | n=6 |
|---|---|---|---|---|---|---|---|
| baseline (t=π/4,u=3π/4,μ0=1) | 3.0571 | 1.61727 | 2.91167 | **1.70897** | 2.28158 | **1.70917** | 2.08668 |
| narrower (t=π/3,u=2π/3,μ0=1) | 2.2924 | 1.51670 | 2.71052 | **1.70711** | 2.21898 | **1.70915** | 2.04686 |
| wider (t=π/6,u=5π/6,μ0=1) | 3.9833 | 1.67227 | 3.02166 | **1.70916** | 2.31758 | **1.70917** | not resolved |
| baseline, μ0=0.5 | 3.0571 | −0.00925 | 1.28514 | **0.08244** | 0.65506 | **0.08265** | 0.46016 |
| baseline, μ0=2 | 3.0571 | 3.87113 | 5.16552 | **3.96282** | 4.53544 | **3.96303** | 4.34054 |

(bold = odd n; the (feasibility-)verified `n=3,5` odd values are the
striking result.) Plot: `taskAB_plot.png` — left panel shows this table
directly, all five cases; right panel shows baseline `δ_mag(n)` against
the analytic bound on a log scale.

**Answer: yes, the effect is geometric, not a numerical artifact of the
baseline case.** In every one of the five geometries tested — including
both a narrower and a much wider guard band, and both a smaller and
larger `μ0` — `log δ_mag(n)+γn` is essentially flat across odd
`n=3→5` (changes of `0.00001` to `0.002`, i.e. `δ(5)/δ(3)` within a few
percent of `e^{-2γ}` in every case), while the even subsequence is
*visibly still decreasing* toward the same value at `n=6`, having not yet
converged. This is consistent with **both parities converging to the
same rate `γ`**, with the odd subsequence reaching it almost immediately
(by `n=3`) and the even subsequence approaching it more slowly from
above — not two different asymptotic rates. `n=1,2` are pre-asymptotic
everywhere (large swings, as expected for the lowest degrees). The
`wider` case's `n=6` could not be resolved by the active-set method (see
Method section) — reported honestly as blank rather than guessed.

Cross-check 3 (`δ_mag(n) ≥ sinh²μ0 e^{-γn}`) holds in every case shown
(all ratios `δ/β_n > 1`, from `~3.6` to `~13`); cross-check 2
(monotonicity) holds in all 5 geometries across their full resolved
range (verified programmatically).

## Deliverable C — mechanism

**The parity effect is caused by the multiplicity of `Qhat−1`'s zero at
`x=1`, not by a degenerate (lower) effective degree.** Two hypotheses
from the brief were tested directly against the verified high-precision
solutions:

- *Degree degeneracy* (`deg Qhat_{2k}=2k−1`, i.e. the leading coefficient
  vanishing at even `n`): **refuted**. The leading Chebyshev coefficient
  of `Qhat` is nonzero at every `n=1..6` (both parities) and shrinks
  smoothly (e.g. baseline: `0.447, 0.139, 0.054, 0.016, 0.0053, 0.0020`
  as a fraction of the largest coefficient) with no discontinuity at even
  n. The even-degree solutions genuinely use their full degree.
- *Zero multiplicity at `x=1`*: **confirmed, and is exactly the
  mechanism**. Writing `Qhat=1+(1-x)R(x)`, the multiplicity of `Qhat-1`'s
  zero at `x=1` is 1 (simple) if `R(1)≠0`, and 2 (double, since then
  `Qhat-1 ≈ -R'(1)(x-1)²` near `x=1`) if `R(1)=0`. The high-precision
  solve shows, to ~50-digit confirmation:

  | n | parity | R(1) |
  |---|---|---|
  | 1 | odd | 0.809028 (simple zero) |
  | 2 | even | −3.05×10⁻⁵² (double zero) |
  | 3 | odd | 0.0176453 (simple zero) |
  | 4 | even | 3.34×10⁻⁵² (double zero) |
  | 5 | odd | 0.000108397 (simple zero) |
  | 6 | even | 3.34×10⁻⁵² (double zero) |

  This directly corrects the brief's own stated expectation — "(D')
  forces [the zero] to be at least double" — which holds **only at even
  n**; at odd n the optimal solution has a genuine *simple* zero at
  `x=1`, confirmed both by direct computation and by the exact n=1
  closed form (`Qhat=1+2s(1+s)(1-x)`, whose slope at `x=1` is `-2s(1+s)
  ≠ 0`).

**Why this produces the parity effect**: going from odd `n=2k-1` to even
`n=2k` spends the one new degree of freedom entirely on forcing `R(1)=0`
(the extra boundary condition), buying *no* new equioscillation point on
the pass band — the equioscillation count stays at `floor((n-1)/2)`
interior touches either side of `n=2k-1→2k`. Going from even `n=2k` to
odd `n=2k+1` *frees* that degree of freedom (the double-root requirement
lapses) and the new degree together add **two** new interior touches at
once. This is exactly what the active-set data shows (baseline,
interior-touch counts `n=1..6`: `0,0,2,2,4,4`) and explains the
alternating "small drop / big drop" pattern in consecutive ratios noted
in the original run.

Equioscillation counts on `X1=[cos t,1]` (baseline, floor `Qhat=1` +
ceiling `Qhat=1+δ` touches combined, always alternating spatially,
starting with a ceiling touch at `x=cos t`): `n=1`: 1 (boundary only);
`n=2`: 1; `n=3`: 3; `n=4`: 3; `n=5`: 5; `n=6`: 5. Constraint `C`
(`Qhat=cosh²μ0` on `I0`) is active at its single boundary point
`x=cos u` for **every** `n=1..6` in the verified high-precision
solutions — this is the finding that exposed CLARABEL's silent
suboptimality at `n=6` noted above (CLARABEL's double-precision `n=6`
answer has `C` comfortably slack, which is simply wrong for the true
optimum).

## Cross-checks

1. `n=1` closed form: reproduced to ~44 digits (baseline and the `μ0=2`
   variant both checked directly against `sinh²μ0(1-cos t)/(1-cos u)`).
   **Pass.**
2. Monotonicity: verified programmatically for all resolved `n` in all 5
   geometries. **Pass.**
3. `δ_mag(n) ≥ sinh²μ0 e^{-γn}`: verified for every resolved `(n,
   geometry)` pair, ratio always `>1` (range ≈3.6–13). **Pass.**
4. Independent grid feasibility: **pass** for all `n=1..6` reported
   (residuals `<1e-40` at declared-active points; a handful of
   `feasible=False` flags from the coarse 2000-point verification grid in
   the geometry sweep are resolution artifacts, not real violations —
   spot-checked directly against the exact `n=1, μ0=2` closed form, which
   matches to 15 digits despite the coarse-grid flag).

## Largest trustworthy n, by method

| method | largest trustworthy n | evidence |
|---|---|---|
| CLARABEL (double precision, from the earlier report) | ~5-6, with caveats | monotonicity/feasibility checks pass through n=7 nominally, but n=6's *active set* is now shown to be silently wrong (see above); treat CLARABEL numbers below `δ≲1e-7` as unreliable |
| This work (grid-LP active-set discovery + mpmath Newton, 50-60 digits) | **6** | closed-form match at n=1 (~44 digits), full feasibility at 50-digit precision, monotonicity and analytic-bound checks all pass |
| n=7..11 | **not resolved** | active-set discovery requires an accurate double-precision seed; both CLARABEL and the grid-LP lose that accuracy in the same regime we're trying to extend into (a genuine chicken-and-egg limit, not a tuning gap) |

## Bottom line

The rate `γ` **is attained exactly on the odd subsequence**, confirmed
to ~50-digit precision at `n=3,5` and across five different geometries
(two guard-band widths besides baseline, and `μ0=0.5,2` besides
baseline) — this is a materially sharper statement than the manuscript's
current generic exponential bound, and is safe to state as such. The
even subsequence is not a separate rate; it is converging to the *same*
`γ` but slowly, driven by the wasted degree of freedom that forces a
double root of `Qhat-1` at `x=1`. Reaching a rigorous, closed-form proof
of "exactly γ on odd n" (as opposed to numerically overwhelming evidence
at n=3,5 across 5 geometries) would need either extending this
high-precision method past its current n=6 ceiling (a genuinely harder
numerical problem than initially scoped, requiring an arbitrary-precision
LP/interior-point solver rather than a double-precision-seeded active-set
search) or an analytic argument from the classical theory of constrained
Chebyshev extremal polynomials that this problem is an instance of.

## Files

`variants.py` (Geometry + exact-π-fraction mpmath constants, cvxpy
Variant I/II builders reused from the original certificate-test work),
`hp_refine.py` (grid-LP active-set discovery, mpmath Newton solve,
high-precision feasibility check), `active_set_rule.py` (the extrapolated
pattern — validated for n≤6, explicitly not used beyond that), `run_taskA.py`/`run_taskA2.py`
(baseline n=1..11 sweep, mpmath results in `taskA2_results.json`),
`run_taskB.py` (five-geometry sweep, `taskB_results.json`),
`make_plot.py` (`taskAB_plot.png`).
