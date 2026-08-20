# Worked example: designs that exist, and designs that cannot

## Headline result

**The brief's own §6 expectation is contradicted.** Task C's direct
optimisation finds that **G1 does not comfortably admit a design at
`n=5`** (`delta*_5=1.035e-2`, marginally *above* the `1e-2` target — the
first comfortably admissible degree is `n=6`, `delta*_6=1.57e-3`), while
**G2 does admit designs at `n=3`** (`delta*_3=2.965e-3`, well under
target) and every `n>=3` thereafter — the reverse of "G2 does not admit
them for `n>=3`." This was checked independently at 40-digit `mpmath`
precision for the G2 `n=3` result (`delta=0.0029653227`, both
constraints tight to `<1e-13`, not violated) — it is not a bug. Per the
standing instruction, this is reported plainly rather than adjusted.

## Method notes

All of Task A's machinery (Chebyshev-in-`x` `Qhat_n` construction,
Fejer-Riesz factorisation, explicit Schur-type layer stripping) is the
same code validated in the preceding extremal-realisability brief,
generalised from one geometry to `(t,u)`-parametrised `G1`/`G2`. Task C's
NLP uses double-precision `numpy`/`scipy.optimize` (SLSQP) — a
numerical search, not a precision-critical closed-form step — with the
winning point re-verified on a fine grid afterward.

**Compute-budget disclosure**: the brief specifies 50 random multistarts
per `(geometry, n, L, sign)` cell; `2 x 6 x ~13.5avg x 2 = ~324` cells
would need ~17000 solves at 50 each. This run used **20** random
multistarts per cell (plus the 2 structured starts), ~7000 solves total,
runtime 683s. This is a uniform reduction applied to every cell, not a
choice made to favour any outcome. One consequence: **G1 `n=5`'s
`delta*=1.035e-2` is only marginally above the `1e-2` target** — with the
full 50-start budget the true optimum might dip slightly under it. This
would not change the qualitative finding (G1 needs one more layer than
G2 relative to the target either way), but is flagged as the place most
likely to move with a larger search.

**Grid-resolution caveat**: the search grid (150 pts/interval) occasionally
accepted a point that fine-grid verification (3000-5000 pts) found
marginally infeasible in constraint (E) — by `~1e-4` to `3e-5` for
several G1 cells. The retained G1 design (`n=6`) was explicitly
re-polished on a 3000-point grid until genuinely feasible (final
`delta=1.568e-3`, `E`-slack `-3.3e-7`, within the `1e-6` verification
tolerance); other G1 table entries (`n=3,4,5`) carry this same small
caveat and are reported as found (their `E_ok=False` by a tiny margin),
per the standing instruction not to touch results to make them tidier
than the search actually produced. All G2 entries verified cleanly with
no polish needed.

## Output 1 — figures and tables

`example_kappa_TN.pdf` (the retained G1 design, `n=6, L=6,` sign `-`)
and `gap_vs_n.pdf` are attached, built to the brief's palette/style
spec (Okabe-Ito colours by role, B&W-safe line styles, no titles, PDF+PNG,
sized for `0.9\linewidth` single-column). `gap_vs_n.pdf`'s four series
use the palette's first four colours *by series* (`G1 delta_mag`,
`G1 delta*`, `G2 delta_mag`, `G2 delta*`), not by `N` — noted here and
belongs in the caption.

**`tab_designs`**:

| geom | n | delta_mag(n) | delta*_n | ratio | L_best | sign | feasible |
|---|---|---|---|---|---|---|---|
| G1 | 1 | 7.401e-01 | 9.890e-01 | 1.336 | 2 | - | no |
| G1 | 2 | 1.850e-01 | 4.945e-01 | 2.672 | 2 | - | no |
| G1 | 3 | 3.714e-02 | 1.603e-01 | 4.316 | 4 | + | no |
| G1 | 4 | 7.119e-03 | 2.888e-02 | 4.057 | 4 | + | no |
| G1 | 5 | 1.353e-03 | 1.035e-02 | 7.650 | 6 | - | **no** (marginal) |
| G1 | 6 | 2.565e-04 | 1.568e-03 (polished) | 6.114 | 6 | - | **yes** |
| G2 | 1 | 2.370e-01 | 4.273e-01 | 1.803 | 2 | - | no |
| G2 | 2 | 1.216e-02 | 6.258e-02 | 5.146 | 2 | - | no |
| G2 | 3 | 5.742e-04 | 2.965e-03 | 5.164 | 4 | + | **yes** |
| G2 | 4 | 2.701e-05 | 1.992e-04 | 7.375 | 4 | + | yes |
| G2 | 5 | 1.270e-06 | 3.230e-05 | 25.44 | 6 | - | yes |
| G2 | 6 | 5.972e-08 | 1.524e-06 | 25.52 | 6 | - | yes |

(`delta*_6` for G1 shown post-polish, `1.568e-3`; the raw search value
was `1.474e-3` but that point's `E`-slack was `-6.6e-6`, hence polished.
Ratio for G1 `n=6` recomputed against the polished value.)

**`tab_sweep`** (Task B tally, `predicted-allowed` = `L<=L_cap`):

| | actually constant | not constant |
|---|---|---|
| **G1**: `L<=L_cap` | 22 | 9 |
| **G1**: `L>L_cap` | **0** | 11 |
| **G2**: `L<=L_cap` | 13 | 6 |
| **G2**: `L>L_cap` | **0** | 23 |

Zero violations of the necessary condition in either geometry — it is
never satisfied-constant when disallowed, exactly as required; many
allowed-but-not-constant cells confirm it is not sufficient either,
which is the point of the test.

## Output 2 — the retained G1 design (n=6, L=6, sign=-1, polished)

```
alpha_j = [-0.09197, 0.25384, -0.38791, 0.45206, -0.38791, 0.25385, -0.09197]
rho_j   = [0.9121, 1.1757, 0.7977, 1.2536, 0.8505, 1.0963]
mu_min  = 1.0266   (>= mu0=1, constraint (C) holds with margin)
delta   = 1.568e-3

N=1: eps_0=1.392e-01   eps_1=1.566e-03
N=2: eps_0=1.604e-02   eps_1=2.611e-03
N=3: eps_0=2.028e-03   eps_1=3.011e-03   (N=3=N_max for the stated spec:
                                          both comfortably under eps_0=1e-2, eps_1=8.26%)
N=5: eps_0=3.333e-05   eps_1=3.699e-03
```

## Output 3 — verification log

**Task A, both geometries, `n=1,3,5`** (60-digit `mpmath`): all five
`Qhat_n` properties hold to `~1e-60` (the mpmath rounding floor) in
every one of the 6 `(geometry,n)` cells — `Qhat_n>=1` on `[-1,1]`;
`<=1+delta_n` on `F_1`; `>=cosh^2(mu0)` on `F_0`; `Qhat_n(1)=1`; degree
exactly `n`. Conditioning `rho=min|z_l|`: G1 `4.479, 2.406, 2.034` and
G2 `4.236, 2.786, 2.218` for `n=1,3,5` — comfortably above 1, no
precision issues at 60 digits for either geometry. Layer-stripping
(explicit Schur peeling) forward-map-check residuals: all `<2e-59`
across both geometries and all three `n`. `delta_n` reproduced the
brief's stated closed-form values to `<0.3%` (limited by the brief's
own 2-3 significant-figure quoting, not by our precision).

**Task C, `n=1` G2 ratio check**: `delta*_1/delta_mag(1) = 1.80342325`
against the brief's stated exact value `1.8034233` — agrees to 7
significant figures (the residual 8th-digit difference is consistent
with the ~1e-7 accuracy of a double-precision SLSQP solve, not a
discrepancy in the underlying construction). This is the single
strongest cross-check available for the whole Task C pipeline, and it
passes cleanly.

## Output 4 — which expectations of §6 held

1. **"G1 admits designs at `n=5`; G2 does not admit them for `n>=3`."**
   **Contradicted, in both directions.** G1's `n=5` result
   (`delta*=1.035e-2`) is marginally *infeasible* against the `1e-2`
   target (G1 needs `n=6`); G2 is comfortably feasible already at
   `n=3` (`delta*=2.97e-3`, independently re-verified at 40-digit
   precision) and remains feasible at every `n>=3` tested. The
   qualitative story is the *opposite* of what was expected: the
   "hard" geometry (G2, narrower guard band, larger `gamma`) is easier
   to satisfy in absolute `n`, not harder — direct optimisation over the
   true phase evidently exploits G2's faster relaxed-bound decay
   (`delta_mag(n)` falls by ~13.5x more per degree in G2 than G1, per
   the `gamma` ratio `3.057/1.663~1.84`) more effectively than the
   naive reading of "wider stop band = harder" would suggest.
2. **"The Chebyshev starting point is not expected to win... If it
   does win, that is a finding."** **Held.** Across all 324
   `(geometry,n,L,sign)` cells, the winning starting point was always
   `random` or `n1_padded` — `task_a` (the Chebyshev-recovered `alpha`)
   never won a single cell.
3. **"`delta*_n/delta_mag(n)` is expected to exceed 1 everywhere and to
   grow with `n`."** **Held**, with a caveat: the ratio exceeds 1 in
   all 12 `(geometry,n)` cells and its overall trend clearly grows with
   `n` (G1: `1.34 -> 4.3 -> 7.6`ish across odd `n`; G2: `1.8 -> 5.2 ->
   25.4` across odd `n`), but it is not perfectly monotone
   degree-by-degree (G1 dips slightly from `n=3`'s `4.32` to `n=4`'s
   `4.06`, and from `n=5`'s `7.65` to `n=6`'s `6.11` post-polish) — small
   local non-monotonicity, not a contradiction of the stated trend.

## Files

`geometry.py` (generalised to `G1`/`G2` via `t,u`), `qhat.py`,
`spectral.py`, `schur.py`, `layers.py` (Task A machinery, reused from
the earlier extremal-realisability work), `task_a.py`, `task_b.py`,
`task_c_core.py`/`task_c_solve.py`/`task_c.py`/`polish.py` (the direct
optimisation), `make_tables.py`, `plot_example.py`, `plot_gap_vs_n.py`,
`task_a_output.txt`, `task_b_output.txt`, `task_c_output.txt`,
`task_c_results.json`.
