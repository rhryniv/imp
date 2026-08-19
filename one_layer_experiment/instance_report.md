# One-layer instance (s=1, baseline t=pi/4, u=3pi/4): report

Built from `transfer_matrix.py`'s matrix machinery (Section 1 of the
brief), not the closed forms — `closed_forms.py` used only as a
cross-check, matching to `~1e-15` (double precision) on this instance.
Grid convergence checked by halving the step (4001 vs 8001 points): max
diff over every reported quantity `1.13e-06`.

## Output 1 — LaTeX table

```latex
\begin{tabular}{c|c|c}
\hline
$N$ & $\epsilon_0(N)$ & $\epsilon_1(N)$ \\
\hline
2  & 0.00624 & 0.5424 \\
8  & 6.76e-11 & 0.5628 \\
16 & 1.62e-21 & 0.5762 \\
\hline
\end{tabular}
```

## Output 2 — scalars

```
alpha  = 0.881374
rho_1  = 2.414214   (= 1+sqrt(2), as expected: e^arcsinh(1) = 1+sqrt(2))
mu_min = 1.529
delta  = 1.172        (max_I1 (Q-1), the ripple parameter of (B))
1 - min_I1 T_env = 0.585786   (min_I1 T_env = 0.414214 = sqrt(2)-1)
```

Task 2 comparison (surrogate `T_N >= 1/(1+N^2 delta)` vs true `eps_1(N)`):

```
N=2:   surrogate eps_1 = 0.8241   true eps_1 = 0.5424   ratio 1.52
N=8:   surrogate eps_1 = 0.9868   true eps_1 = 0.5628   ratio 1.75
N=16:  surrogate eps_1 = 0.9967   true eps_1 = 0.5762   ratio 1.73
```

(In terms of the `T_N` floor itself rather than `eps_1`, the surrogate's
looseness is far more dramatic: at `N=2` it predicts `T_N>=0.176` against
a true floor of `0.458` — 2.6x pessimistic; at `N=16` it predicts
`T_N>=0.0033` against a true floor of `0.424` — over 100x pessimistic.
The gap widens with `N` because the surrogate uses the crude
`N`-uniform bound `|U_{N-1}|<=N`, while the true `kappa_B` stays
comfortably inside `(-1,1)` on `I_1`, so the actual `|U_{N-1}(kappa_B)|`
is far below `N`.)

## Output 3 — figure

`onelayer_kappa_TN.pdf` (vector) / `.png`. Sanity checks run before
saving, all passed: `kappa_B(0)=1` (diff `<1e-10`); `T_N(0)=1` for
`N=2,8,16` (diff `<1e-10` each); `T_N>=T_env` pointwise inside both open
bands; `T_N<=1` everywhere. Sized for `0.9\linewidth` in a two-column
layout (3.4in figure width, 8-9pt fonts at that size); `N=2,8,16`
distinguished by line style (solid/dashed/dotted), not color alone.

## Output 4 — summary

Every quantity in Task 1 reproduces the earlier run to within the
grid-resolution level (`mu_min` diff `2.9e-5`; `max_I0_T_N` diffs
`2.4e-6` to `4.4e-24`; `min_I1_T_env` diff `4.4e-7`, matching the
closed-form `sqrt(2)-1` exactly) — **no disagreement found**. The Task 2
surrogate bound is confirmed *valid* (it never exceeds the true
`eps_1(N)`) but **substantially loose**, and increasingly so with `N`:
in `eps_1` terms it overstates the shortfall by 1.5-1.75x, but in the
underlying `T_N`-floor terms the looseness grows from ~2.6x at `N=2` to
over 100x at `N=16`, because it bounds `|U_{N-1}(kappa_B)|` by the
crude `N`-uniform ceiling rather than using `kappa_B`'s actual (safely
interior) value on `I_1` — the same gap identified earlier in this
project's history as the reason the old `s_0`-based pass-band gate was
replaced.
