"""Instance 1 lineage: I_0=[2.8,pi], I_1=[0,0.3] -- same intervals as
instance1_alt6_early_stop_scan.py (which found n=3 admissible in 12.6s
via early stop, having also computed n=2). This variant reports EVERY
degree n=1,2,3 individually (early stop disabled, degree_scan_stage3
directly) with its own design plot, not just the retained one.

M=25 (n_grid_B=n_grid_C=25). eps_0=eps_1=1e-2, N_max=60
(mu0=log(4/eps0)/(2*N_max)~=0.05).

Run from the repository root: python3 examples/instance1_alt6_n1to3_scan.py
"""
import json
import time

import numpy as np

from scattering.driver import (
    degree_scan_stage3, retained_degree_record, save_records_csv, save_records_latex, mu0_from_N_max,
)
from scattering.plotting import plot_design_from_record, plot_bound_vs_achieved

I0 = [(2.8, np.pi)]
I1 = [(0.0, 0.3)]
EPS0 = EPS1 = 1e-2
N_MAX = 60
N_RANGE = range(1, 4)
N_VALUES = (20, 40, 60, 80, 100)
N_GRID_B = 25
N_GRID_C = 25


def _fmt(x):
    return "None" if x is None else f"{x:.6e}"


def main():
    t0 = time.time()
    records = degree_scan_stage3(N_RANGE, I0, I1, N_MAX, EPS0, EPS1, N_values=N_VALUES,
                                  n_grid_B=N_GRID_B, n_grid_C=N_GRID_C)
    elapsed = time.time() - t0
    print(f"scan finished in {elapsed:.1f}s, {len(records)} degree(s) computed, M={N_GRID_B}/{N_GRID_C}")

    for r in records:
        print(f"n={r.n:2d} status={r.status:20s} underline_delta={_fmt(r.underline_delta):>14} "
              f"delta_achieved={_fmt(r.delta_achieved):>14} kappa_min={_fmt(r.kappa_min):>14} "
              f"max_kappa_B={_fmt(r.max_kappa_B):>14} admissible={r.admissible} "
              f"sigma_star={r.sigma_star} time_direct_s={r.time_direct_s:.1f}")

    print()
    retained = retained_degree_record(records)
    if retained is None:
        print("No degree in n=1..3 is admissible (rule 4).")
    else:
        print(f"Retained (least admissible n): n={retained.n}")

    save_records_csv(records, "examples/instance1_alt6_n1to3_scan.csv")
    save_records_latex(records, "examples/instance1_alt6_n1to3_scan_table.tex")
    with open("examples/instance1_alt6_n1to3_scan.json", "w") as f:
        json.dump([r.to_json_dict() for r in records], f, indent=2)

    plot_bound_vs_achieved(records, retained=retained, savepath="examples/instance1_alt6_n1to3_bound_vs_achieved.png")

    mu0 = mu0_from_N_max(EPS0, N_MAX)
    for r in records:
        if r.status == "optimal" and r.alpha is not None:
            plot_design_from_record(r, I0, I1, N_values=(1, 3, 5), mu0=mu0,
                                     savepath=f"examples/instance1_alt6_n{r.n}_design.png")
            print(f"saved design plot for n={r.n}")
        else:
            print(f"n={r.n}: status={r.status}, no design to plot")

    print("\nSaved: examples/instance1_alt6_n1to3_scan.{csv,json}, instance1_alt6_n1to3_scan_table.tex, "
          "instance1_alt6_n1to3_bound_vs_achieved.png, instance1_alt6_n{1,2,3}_design.png (where optimal)")


if __name__ == "__main__":
    main()
