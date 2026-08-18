"""Instance 1 lineage: I_0=[3*pi/4,pi], I_1=[0,pi/4] (RH's own choice) --
I_1 hugs theta=0 (kappa_B=1 there exactly, a closed gap by the matching
constraint; allowed under the manuscript revision, rem:k0/Task 6a) and
I_0 abuts theta=pi (structurally an open gap by construction, Remark
rem:endpoints, L=2n fixed, unless the design closes it) -- the same
"pass near 0, stop near pi" placement that worked well in
instance1_alt4_coarse_grid_early_stop_scan.py.

RH's own request: only n=1..5 (not the usual n=2..16), M=25
(n_grid_B=n_grid_C=25), and EVERY degree in that range reported --
stop_at_first_admissible is deliberately OFF here (degree_scan_stage3
directly, not find_retained_design_stage3) so n=1..5 are all computed
regardless of whether an earlier one is already admissible.

eps_0=eps_1=1e-2, N_max=60 (mu0=log(4/eps0)/(2*N_max)~=0.05).

Run from the repository root: python3 examples/instance1_alt5_n1to5_scan.py
"""
import json
import time

import numpy as np

from scattering.driver import (
    degree_scan_stage3, retained_degree_record, save_records_csv, save_records_latex, mu0_from_N_max,
)
from scattering.plotting import plot_design_from_record, plot_bound_vs_achieved

I0 = [(3 * np.pi / 4, np.pi)]
I1 = [(0.0, np.pi / 4)]
EPS0 = EPS1 = 1e-2
N_MAX = 60
N_RANGE = range(1, 6)
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
    print(f"scan finished in {elapsed:.1f}s, {len(records)} degree(s) computed "
          f"(of {len(list(N_RANGE))} in the requested range), M={N_GRID_B}/{N_GRID_C}")

    for r in records:
        print(f"n={r.n:2d} status={r.status:20s} underline_delta={_fmt(r.underline_delta):>14} "
              f"delta_achieved={_fmt(r.delta_achieved):>14} kappa_min={_fmt(r.kappa_min):>14} "
              f"max_kappa_B={_fmt(r.max_kappa_B):>14} admissible={r.admissible} "
              f"time_direct_s={r.time_direct_s:.1f}")

    print()
    retained = retained_degree_record(records)
    if retained is None:
        print("No degree in the requested range is admissible (rule 4).")
    else:
        print(f"Retained (least admissible n): n={retained.n}, delta_achieved={retained.delta_achieved}, "
              f"sigma_star={retained.sigma_star}, max_kappa_B={retained.max_kappa_B}")

    save_records_csv(records, "examples/instance1_alt5_scan.csv")
    save_records_latex(records, "examples/instance1_alt5_scan_table.tex")
    with open("examples/instance1_alt5_scan.json", "w") as f:
        json.dump([r.to_json_dict() for r in records], f, indent=2)

    plot_bound_vs_achieved(records, retained=retained, savepath="examples/instance1_alt5_bound_vs_achieved.png")

    plot_source = retained if retained is not None else next(
        (r for r in reversed(records) if r.status == "optimal" and r.alpha is not None), None)
    if plot_source is not None:
        plot_design_from_record(plot_source, I0, I1, N_values=(1, 3, 5), mu0=mu0_from_N_max(EPS0, N_MAX),
                                 savepath="examples/instance1_alt5_design.png")
        print(f"design plot uses n={plot_source.n} "
              f"({'retained' if retained is not None else 'best available, none admissible'})")

    print("\nSaved: examples/instance1_alt5_scan.{csv,json}, instance1_alt5_scan_table.tex, "
          "instance1_alt5_bound_vs_achieved.png, instance1_alt5_design.png")


if __name__ == "__main__":
    main()
