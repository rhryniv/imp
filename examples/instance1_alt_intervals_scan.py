"""Instance 1 rerun with an alternate interval choice (RH's own choice,
not spec Sec. 7's original I_0=[pi/6,pi/4], I_1=[pi/2,3*pi/4] -- see
instance1_scan.py for that run and its committed results): I_0=[1.5,1.9],
I_1=[0.1,0.5] (m_0=m_1=1, pass band now precedes the stop band in theta),
eps_0=eps_1=1e-2, N_max=60 (manuscript revision: N_max is now the
top-level input in place of mu_0 -- see driver.py's own module docstring
-- chosen so the derived mu0=log(4/eps0)/(2*N_max) ~= 0.05, the original
mu_0 this run used), n=2..16.

Produces the same Sec. 6 emissions table (CSV + LaTeX table body) and
deliverable-3 plots as instance1_scan.py, under an "_alt" suffix so
neither run's output overwrites the other's.

Run from the repository root: python3 examples/instance1_alt_intervals_scan.py
"""
import json
import time

import numpy as np

from scattering.driver import (
    degree_scan_stage3, retained_degree_record, save_records_csv, save_records_latex, mu0_from_N_max,
)
from scattering.plotting import plot_design_from_record, plot_bound_vs_achieved

I0 = [(1.5, 1.9)]
I1 = [(0.1, 0.5)]
EPS0 = EPS1 = 1e-2
N_MAX = 60
N_RANGE = range(2, 17)
N_VALUES = (20, 40, 60, 80, 100)


def _fmt(x):
    return "None" if x is None else f"{x:.6e}"


def main():
    t0 = time.time()
    records = degree_scan_stage3(N_RANGE, I0, I1, N_MAX, EPS0, EPS1, N_values=N_VALUES)
    print(f"scan finished in {time.time() - t0:.1f}s")

    for r in records:
        print(f"n={r.n:2d} status={r.status:20s} underline_delta={_fmt(r.underline_delta):>14} "
              f"delta_achieved={_fmt(r.delta_achieved):>14} admissible={r.admissible} "
              f"s_0={_fmt(r.s_0):>14} dual={r.dual_status} time_direct_s={r.time_direct_s:.1f}")

    retained = retained_degree_record(records)
    print()
    if retained is None:
        print("No degree in the scanned range is admissible (rule 4).")
    else:
        print(f"Retained (least admissible n): n={retained.n}, delta_achieved={retained.delta_achieved}, "
              f"sigma_star={retained.sigma_star}")

    save_records_csv(records, "examples/instance1_alt_scan.csv")
    save_records_latex(records, "examples/instance1_alt_scan_table.tex")
    with open("examples/instance1_alt_scan.json", "w") as f:
        json.dump([r.to_json_dict() for r in records], f, indent=2)

    plot_bound_vs_achieved(records, retained=retained, savepath="examples/instance1_alt_bound_vs_achieved.png")

    plot_source = retained if retained is not None else next(
        (r for r in reversed(records) if r.status == "optimal" and r.alpha is not None), None)
    if plot_source is not None:
        plot_design_from_record(plot_source, I0, I1, N_values=(1, 3, 5), mu0=mu0_from_N_max(EPS0, N_MAX),
                                 savepath="examples/instance1_alt_design.png")
        print(f"design plot uses n={plot_source.n} "
              f"({'retained' if retained is not None else 'best available, none admissible'})")

    print("\nSaved: examples/instance1_alt_scan.{csv,json}, instance1_alt_scan_table.tex, "
          "instance1_alt_bound_vs_achieved.png, instance1_alt_design.png")


if __name__ == "__main__":
    main()
