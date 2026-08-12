"""Instance 1, another alternate interval choice (RH's own choice):
I_0=[1.5,2.0], I_1=[0.1,0.5] -- close to instance1_alt_intervals_scan.py's
I_0=[1.5,1.9], I_1=[0.1,0.5] (which found n=3 admissible under the
pre-manuscript-revision s_0 gate) but with I_0 widened to [1.5,2.0], and
run here with the new N_max-derived gate plus the
stop_at_first_admissible early exit (driver.find_retained_design_stage3)
-- see driver.degree_scan_stage3's own docstring for why the early stop
is lossless, not an approximation.

eps_0=eps_1=1e-2, N_max=60 (mu0=log(4/eps0)/(2*N_max)~=0.05).

Run from the repository root: python3 examples/instance1_alt2_early_stop_scan.py
"""
import json
import time

import numpy as np

from scattering.driver import find_retained_design_stage3, save_records_csv, save_records_latex, mu0_from_N_max
from scattering.plotting import plot_design_from_record, plot_bound_vs_achieved

I0 = [(1.5, 2.0)]
I1 = [(0.1, 0.5)]
EPS0 = EPS1 = 1e-2
N_MAX = 60
N_RANGE = range(2, 17)
N_VALUES = (20, 40, 60, 80, 100)


def _fmt(x):
    return "None" if x is None else f"{x:.6e}"


def main():
    t0 = time.time()
    retained, records = find_retained_design_stage3(N_RANGE, I0, I1, N_MAX, EPS0, EPS1, N_values=N_VALUES)
    elapsed = time.time() - t0
    print(f"scan finished in {elapsed:.1f}s, {len(records)} degree(s) computed "
          f"(of {len(list(N_RANGE))} in the full range)")

    for r in records:
        print(f"n={r.n:2d} status={r.status:20s} underline_delta={_fmt(r.underline_delta):>14} "
              f"delta_achieved={_fmt(r.delta_achieved):>14} kappa_min={_fmt(r.kappa_min):>14} "
              f"max_kappa_B={_fmt(r.max_kappa_B):>14} admissible={r.admissible} "
              f"time_direct_s={r.time_direct_s:.1f}")

    print()
    if retained is None:
        print("No degree in the scanned range is admissible (rule 4) -- full range was scanned.")
    else:
        print(f"Retained (least admissible n): n={retained.n}, delta_achieved={retained.delta_achieved}, "
              f"sigma_star={retained.sigma_star}, max_kappa_B={retained.max_kappa_B}")

    save_records_csv(records, "examples/instance1_alt2_early_stop_scan.csv")
    save_records_latex(records, "examples/instance1_alt2_early_stop_scan_table.tex")
    with open("examples/instance1_alt2_early_stop_scan.json", "w") as f:
        json.dump([r.to_json_dict() for r in records], f, indent=2)

    plot_bound_vs_achieved(records, retained=retained,
                            savepath="examples/instance1_alt2_early_stop_bound_vs_achieved.png")

    plot_source = retained if retained is not None else next(
        (r for r in reversed(records) if r.status == "optimal" and r.alpha is not None), None)
    if plot_source is not None:
        plot_design_from_record(plot_source, I0, I1, N_values=(1, 3, 5), mu0=mu0_from_N_max(EPS0, N_MAX),
                                 savepath="examples/instance1_alt2_early_stop_design.png")
        print(f"design plot uses n={plot_source.n} "
              f"({'retained' if retained is not None else 'best available, none admissible'})")

    print("\nSaved: examples/instance1_alt2_early_stop_scan.{csv,json}, instance1_alt2_early_stop_scan_table.tex, "
          "instance1_alt2_early_stop_bound_vs_achieved.png, instance1_alt2_early_stop_design.png")


if __name__ == "__main__":
    main()
