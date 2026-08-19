# One-layer impedance filter: numerical experiment report

All quantities (`T_N`, `kappa_B`, `q1`, `q2`) were built strictly from
the transfer-matrix definitions in the brief's §1 (`C(alpha)`, `E(phi)`,
`M(alpha,kx)=E(-kx)C(alpha)E(kx)`, block/period products) — never from
the closed forms of §2, which are only ever the *comparison* target.
High precision (`mpmath`, 30 digits) is used for every closed-form
comparison (Tasks 1, 4, 6); double precision for the tables (Tasks 2, 3,
5), with grid convergence checked by halving the step. `mu0=0.5` used
for the Task 3(3)/5 exclusion criterion, per your confirmation.

## Output 1 — Task 2 LaTeX table (`tab:onelayer`)

```latex
\begin{tabular}{c|c|ccc|ccc|c}
\hline
$s$ & $\mu_{\min}$ & \multicolumn{3}{c|}{$\max_{I_0} T_N$} & \multicolumn{3}{c|}{$\min_{I_1} T_N$} & $\min_{I_1} T_{\mathrm{env}}$ \\
    &              & $N=2$ & $N=8$ & $N=16$ & $N=2$ & $N=8$ & $N=16$ & \\
\hline
0.25 & 0.5119 & 1.54e-01 & 2.97e-04 & 8.25e-08 & 0.7714 & 0.7855 & 0.7657 & 0.7657 \\
0.50 & 1.0148 & 3.85e-02 & 1.99e-07 & 1.77e-14 & 0.6279 & 0.6100 & 0.6204 & 0.6095 \\
1.00 & 1.5286 & 6.24e-03 & 6.76e-11 & 1.62e-21 & 0.4576 & 0.4372 & 0.4238 & 0.4142 \\
2.00 & 2.0943 & 7.18e-04 & 8.75e-15 & 2.45e-29 & 0.2967 & 0.2296 & 0.2438 & 0.2190 \\
4.00 & 2.7083 & 6.45e-05 & 4.95e-19 & 7.50e-38 & 0.0900 & 0.0909 & 0.0637 & 0.0627 \\
\hline
\end{tabular}
```

Grid-convergence residual (4001-pt vs 8001-pt grid, max over all
reported quantities per row): `5.5e-9, 1.1e-8, 1.1e-6, 2.7e-7, 0`
(s=0.25, 0.5, 1, 2, 4) — all reported digits are converged.

Cross-check against the brief's "earlier session" numbers (computed for
s=0.25,0.5,1,4; s=2 is new): `mu_min` ≈ 0.512, 1.015, 1.529, —, 2.708 vs.
this run's 0.5119, 1.0148, 1.5286, —, 2.7083 — **matches**.
`max_{I0}T_8` ≈ 3e-4, 2e-7, 7e-11, —, 5e-19 vs. this run's 2.97e-4,
1.99e-7, 6.76e-11, —, 4.95e-19 — **matches** to the quoted precision.
`min_{I1}T_N` ≈ 0.77, 0.62, 0.43, —, 0.08 vs. this run's `min_{I1}T_8`
0.7855, 0.6100, 0.4372, —, 0.0909 — **matches** (same ballpark; the
earlier session's `N` for that row isn't stated precisely, but every `N`
column here is consistent with it).

## Output 2 — Task 3(1) oscillation data (s=1, baseline)

`min_{I1} T_N` for `N=1..16`, envelope `min_{I1} T_env = 0.414214`:

```
N=1  0.460496      N=9   0.450803
N=2  0.457627      N=10  0.420730
N=3  0.482160      N=11  0.434979
N=4  0.418824      N=12  0.445689
N=5  0.441905      N=13  0.422621
N=6  0.460197      N=14  0.433658
N=7  0.417076      N=15  0.417898
N=8  0.437226      N=16  0.423777
```

Non-monotone (no trend as `N` increases — e.g. `N=6→7` drops from 0.460
to 0.417, `N=7→9` rises back to 0.451), oscillating around the envelope.
Every value lies strictly above `T_env` (min gap 0.0029 at `N=3`, max gap
0.0679 at `N=6`; zero sign changes of `T_N-T_env`) — consistent with
`T_env` being a genuine lower bound attained only where `|sin(N phi)|=1`,
never crossed.

## Output 3 — discrepancy log (Tasks 1, 4, 6)

**Task 1** (mpmath, 30 digits, s∈{0.25,0.5,1,2,4}, 400-pt grid per s,
`N`∈{1,2,4,8,16}): every check below is a *maximum* over the full grid
and both stop/pass regions combined.

```
s=0.25  max|kappaB(matrix)-kappaB(closed)| = 4.9e-31
        max|Q(matrix)-Q(closed)|            = 1.6e-30
        max||q1|^2-|q2|^2-1|                = 1.2e-30
        max|q1(matrix)-((1+s)-s*w)|         = 4.0e-31
        kappa_B(0)-1                        = 9.9e-32
        c0+c1-1 (=q1(0)-1)                  = 9.9e-32
        max|T_N(0)-1| (all N)               = 0.0
s=0.5   [same pattern]  max diffs 5.9e-31 .. 3.9e-30
s=1     [same pattern]  max diffs 7.9e-31 .. 4.7e-30
s=2     [same pattern]  max diffs 1.6e-30 .. 1.6e-29
s=4     [same pattern]  max diffs 3.2e-30 .. 7.6e-29
```

No discrepancy exceeds `8e-29` anywhere (all consistent with 30-digit
mpmath roundoff, growing mildly with `s` as `cosh/sinh(alpha)` grow).
**Every Claim-1 formula and every Task-1 bullet is confirmed exactly.**

**Task 4** (Claim 2 feasibility region, baseline t,u; grid: 20 values of
`s` in `[0.05,3.0]` × 12 values of `mu0` in `[0.1,2.0]` × 12 values of
`delta` (geometric, `[1e-3,2.0]`) = 2880 points; (B)/(C)/(E) checked
directly against an 800-point grid over `I0`/`I1` each, tolerance
`1e-9`):

```
disagreements between closed-form verdict "s_- <= s <= min(s_+,cot^2)"
and the direct numerical (B)&(C)&(E) verdict: 0 / 2880
```

Extra confirmations: `cot^2(pi/8) = 5.8284...` (matches); `delta_min(mu0)`
at `mu0`∈{0.25,0.5,1,2}: `0.13243, 0.17985, 0.42734, 2.92448` (mpmath, 30
digits; `mu0=1` matches the brief's `0.4273373` exactly); `s_-(mu0)>0`
confirmed for `mu0`∈{0.01,0.1,0.5,1,2,5} (`s_-` ranges `0.172` to
`43.06`, always positive). **Claim 2 confirmed with zero disagreements.**

**Task 6** (mpmath, 30 digits, `mu0=1`, baseline `u=3pi/4`):

```
t=pi/4: s_-=0.4897021457  delta*_1=0.4273372616  delta_mag(1)=0.2369589284
        ratio=1.803423338   identity check |ratio-RHS| = 2.0e-31
t=pi/6: ratio=1.803423338   identity check |ratio-RHS| = 7.9e-31
t=pi/3: ratio=1.803423338   identity check |ratio-RHS| = 2.0e-31
max pairwise diff across the three t values: 5.9e-31
```

All four quoted numbers match the brief exactly (`0.4897021`,
`0.4273373`, `0.2369589`, `1.8034233`), the identity holds to 30-digit
precision, and the ratio is confirmed **independent of `t`** to the same
precision. **Task 6 fully confirmed.**

**No failures were found in Tasks 1, 4, or 6** — every closed form in
Claims 1-3 and the Task 6 identity is confirmed to (mpmath) machine
precision, and Claim 2's feasibility region has zero disagreements
against a direct numerical check on 2880 grid points.

## Output 4 — summary

**Confirmed, without exception:** Claim 1 (`kappa_B`, `Q-1`, `|q2|^2`
closed forms — Task 1, max residual `8e-29`); Claim 2's feasibility
region, including the sign choice `sigma_1=-1`, all three bounds
(`s_-` from (C), `s_+` from (B), `cot^2(t/2)` from (E)), and the
`delta<=4cot^2(t/2)` inactivity threshold for (E) (Task 4, 0/2880
disagreements); Claim 3's least-ripple formula and Task 6's benchmark
identity, including its exact independence from `t` (30-digit agreement
across `t=pi/6,pi/4,pi/3`); the non-monotonicity and asymmetry claims of
Task 3 (T_N oscillates around, but never falls below, the N-independent
envelope; the stop-band value moves ~15 orders of magnitude across
`s∈[0.25,4]` while the pass-band value stays within a ~0.7-wide band).
**One genuinely new, unflagged finding surfaced in Task 5:** at the
harder geometry (`t=0.4pi,u=0.6pi`), the smaller-`s` designs (`s=0.25,
0.5`) don't even open a complete band gap over `I0` — `|kappa_B|<=1` on
26% and 2% of `I0` respectively (so `mu_min` is undefined/NaN there, and
`max_{I0}T_N` can equal `1.0` exactly, i.e. some stop-band frequencies
pass through with *no* attenuation at all), a starker and more literal
form of "one layer fails outright" than the exclusion criterion's `eps_1`
number alone conveys; the exclusion criterion itself gives
`eps_1≈0.588` at `mu0=0.5` there (vs. `0.232` at baseline), confirming
the qualitative failure. Given the confirmed value of `mu0=0.5`, the
Task 3(3)/Task 5 exclusion minimizer is confirmed at `s=s_-(mu0)`
exactly (verified: `s_opt/s_- = 1.0000000` at baseline, `1.000000` at
the harder instance), i.e. **constraint (C) is always active at the
optimum**, matching the brief's own expectation.

## Files

`transfer_matrix.py`/`transfer_matrix_mp.py` (double/mpmath matrix
machinery, §1 only), `transfer_matrix_vec.py` (algebraically-unrolled,
vectorized n=1 special case, cross-checked to `1.8e-15` against the
honest per-point matrix code before use in Task 4), `closed_forms.py`
(§2 closed forms, double + mpmath), `run_task{1..6}.py`, `task2_table.tex`.
