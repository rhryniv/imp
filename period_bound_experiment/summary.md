# Exhaustive enumeration of the period L — summary

Convention note (per the corrected `lem:period-bound`): a block has $n$
layers and $n+1$ contrasts; period $L=n+d$. All of what follows uses
this convention throughout — `n` is layer count, not contrast count.

## 1. Gate results

| Gate | Check | Result | Pass/Fail |
|---|---|---|---|
| G1 | n=1,d=1,σ=-1: α=(0.31118,-0.31118), δ=0.064437 | α₀=0.311182478501175, δ=0.064437080100224 | **PASS** |
| G2 | n=3,d=1,σ=+1 (random+antisym starts only): δ=4.6722e-4, α matches published seed to 5 d.p. | δ=0.0004672221733226, α_full=(0.16227979,-0.39890963,0.39890964,-0.16227980) — exact match to the target ±(0.16228,-0.39891,0.39891,-0.16228) up to the problem's global sign-flip degeneracy (α→-α leaves p1, hence Q and κ, algebraically invariant — verified by induction on the recursion) | **PASS** |
| G3 | Σα=0 and Σc=1 to 1e-12, on G1's and G2's optima | Σα=0.0 (both), Σc-1 = -2.2e-16 (G1), -5.6e-16 (G2) | **PASS** |

**GATES OVERALL: PASS.** Proceeded to the full sweep.

## 2. δ vs d, both signs, per n

Infeasible cells marked "—"; the minimising d per n is **bold**.
δ values are `delta_fine` (Section 3's fine-grid re-evaluation), not the
solver's coarse-grid value.

### n=1

| d | L | σ=+1 | σ=-1 |
|---|---|---|---|
| **1** | 2 | — | **6.443708e-02** |
| 2 | 3 | — | — |
| 3 | 4 | 3.328531e-01 | — |
| 4 | 5 | — | — |
| 5 | 6 | — | — |
| 6 | 7 | — | — |
| 7 | 8 | — | — |
| 8 | 9 | — | — |
| 9 | 10 | — | — |
| 10 | 11 | — | — |
| 11 | 12 | — | — |

### n=3

| d | L | σ=+1 | σ=-1 |
|---|---|---|---|
| **1** | 4 | **4.672222e-04** | — |
| 2 | 5 | — | — |
| 3 | 6 | — | 2.277343e-03 |
| 4 | 7 | — | — |
| 5 | 8 | 2.629024e+00 (fragile — see §6) | — |
| 6 | 9 | — | — |
| 7 | 10 | — | — |
| 8 | 11 | — | — |
| 9 | 12 | — | — |
| 10 | 13 | — | — |
| 11 | 14 | — | — |

### n=5

| d | L | σ=+1 | σ=-1 |
|---|---|---|---|
| **1** | 6 | — | **2.790590e-06** |
| 2 | 7 | — | — |
| 3 | 8 | — | — |
| 4 | 9 | — | — |
| 5 | 10 | — | — |
| 6 | 11 | — | — |
| 7 | 12 | — | — |
| 8 | 13 | — | — |
| 9 | 14 | — | — |
| 10 | 15 | — | — |
| 11 | 16 | — | — |

Every cell not listed as feasible above was searched with the full start
budget (200 random + 50 antisymmetric, plus the n=3 known-block seed and
up to 2 continuation seeds where available — see `config.json`) and
**no start converged to a fine-grid-feasible point**. This means "no
feasible point was found within the search budget," not a proof of
infeasibility — the search is not a certificate.

## 3. Headline answer: is d=1 optimal?

**Yes, for every n tested (1, 3, 5), argmin_d δ = 1** — the previously
reported optima survive.

| n | best δ at d=1 | best δ at any other feasible d | margin (other/d=1) |
|---|---|---|---|
| 1 | 6.443708e-02 | 3.328531e-01 (d=3, σ=+1) | 5.17× worse |
| 3 | 4.672222e-04 | 2.277343e-03 (d=3, σ=-1); 2.629024e+00 (d=5, σ=+1, fragile) | 4.87× worse (d=3); 5628× worse (d=5) |
| 5 | 2.790590e-06 | *(no other feasible d found)* | no competitor found |

d=1 is not merely the best among a close field — at n=1 and n=3 it beats
the next feasible d by a wide margin (~5×), and at n=5 it is the only
feasible d found at all.

## 4. Tightness of the a posteriori cap

$d_{\rm cap}=4\beta_J/|I_0|$, evaluated at each found optimum, against the
unconditional cap $2\pi/|I_0|=12$ (so d must be $\le 11$, matching the
enumerated range) and against the d actually optimal:

| n | d (optimum) | σ | δ | β_J | d_cap (a post.) | unconditional cap | slack (cap / d) |
|---|---|---|---|---|---|---|---|
| 1 | 1 | -1 | 6.4437e-02 | 0.500818 | 3.826 | 12 | 3.83× |
| 1 | 3 | +1 | 3.3285e-01 | 0.949794 | 7.256 | 12 | 2.42× |
| 3 | 1 | +1 | 4.6722e-04 | 0.902537 | 6.895 | 12 | 6.89× |
| 3 | 3 | -1 | 2.2773e-03 | 1.234804 | 9.433 | 12 | 3.14× |
| 3 | 5 | +1 | 2.6290e+00 | 1.548683 | 11.831 | 12 | 2.37× |
| 5 | 1 | -1 | 2.7906e-06 | 1.194249 | 9.123 | 12 | 9.12× |

Every observed $d_{\rm cap}$ is comfortably below the unconditional cap 12
(consistent with $\beta_J\le\pi/2$ always, so $4\beta_J/|I_0|\le
2\pi/|I_0|$ identically — the a posteriori form can never be looser than
the unconditional one, and the n=3,d=5 case at 11.831 is the closest any
optimum came to that theoretical ceiling). At the three **optimal** (d=1)
points, the a posteriori cap has substantial slack — 3.8× to 9.1× the
actual optimal d — so on this evidence the lemma correctly bounds the
admissible range but is far from tight at the actual optimum; it becomes
much tighter (2.4×-3.1× slack) at the higher-d feasible points that were
found, suggesting the bound sharpens as δ grows and the design becomes
more marginal.

## 5. Feasible at d≥2 but infeasible at d=1 (or vice versa)?

Checked per (n, σ) pair, since sign and d cannot be decoupled (Section 7
guard):

- **(n=1, σ=+1): infeasible at d=1, feasible at d=3.** The only feasible
  σ=+1 point for n=1 is at d=3, not d=1.
- **(n=3, σ=-1): infeasible at d=1, feasible at d=3.** Same pattern: the
  only feasible σ=-1 point for n=3 is at d=3.
- No case was found feasible at d=1 but infeasible at every d≥2 with the
  *same* sign — every (n,σ) pair with a feasible d=1 point also had a
  further feasible point at higher d for at least one n (n=3, σ=+1, at
  d=5) or none checked further (n=5, σ=-1, where no d≥2 point was found
  for either sign).

So yes: two genuine instances of "d=1 infeasible, d≥2 feasible" were
found, both for the sign that does *not* give the case's global optimum
— i.e. flipping σ can open up feasibility at a different, always worse,
d. This does not change the headline answer (the overall best δ per n is
always at d=1), but it does mean d=1 infeasibility for one sign is not
evidence of d=1 infeasibility for the design as a whole.

## 6. A flagged low-confidence result: n=3, d=5, σ=+1

Only 1 of 251 starts converged to a fine-grid-feasible point at this
cell (all others either failed to converge or landed infeasible). The
resulting δ=2.629 is enormous compared to every other feasible cell
(next-worst is 3.3e-1) and its κ_min on I0 is 6.395 — wildly non-binding
compared to cosh(μ0)=1.053 at the constraint. All Section 3 fine-grid
checks pass cleanly (min_Q, κ_max, κ_min all well inside tolerance), so
this is not a numerical-precision artifact in the sense of Section 7 —
it is a genuine feasible point, just an extremely poor one, most likely
an isolated feasible basin far from any reasonable local minimum of δ
that one lucky start happened to land in. Reported as found rather than
discarded, but it should not be read as "the" optimal n=3,d=5 design —
with only 1/251 starts finding *any* feasible point here, the search is
under-resourced for this cell and a better (lower-δ) point likely exists
undiscovered within the same feasible region.

## 7. Section 7 traps

- **Conditioning**: the Q-via-autocorrelation vs Q-via-direct-evaluation
  cross-check was run at every reported optimum (spot-checked every
  500th fine-grid node). Largest disagreement observed: 2.10e-13 (n=3,
  d=5 case), all others ≤9e-16. All far below the 1e-10 reporting
  threshold — no conditioning issue encountered, consistent with
  staying far from the σ_slack~10^4 regime the trap describes.
- **Sign enumeration**: σ was swept independently at every d (never
  carried across d) — see Section 5's findings, which depended on doing
  exactly this.

## 8. A numerical-stability deviation, disclosed

Bounds |alpha_j|<=20 and 0<=delta<=10 were added to the SLSQP call (not
specified in the brief) after unbounded alpha excursions during
finite-difference Jacobian steps produced float64 cosh/sinh overflow at
n=5, stalling convergence and making n=5 cases run ~25x slower than n=1
(55s vs 2.2s per case, mostly on non-converging starts burning the full
500-iteration budget). The bounds are far outside the region any genuine
optimum occupies (all found optima have |alpha_j|<2.3) and did not
change either gate's outcome. See config.json for the exact values.
