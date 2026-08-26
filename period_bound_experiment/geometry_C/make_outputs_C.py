"""results_C.csv per spec Sec 7(a)."""
from __future__ import annotations

import csv
import json

with open("results_C_checkpoint.json") as f:
    R = json.load(f)

BASE_COLS = ["n", "d", "L", "sigma", "feasible", "delta", "delta_per_I1_component",
             "alpha", "rho", "A_minus1", "antisym_residual", "sym_residual",
             "p_per_component", "a", "e", "kappa_min_I0", "beta",
             "max_abs_kappa_I1_excluding_zero", "n_starts_converged", "n_starts_total",
             "best_start_type", "wall_time_s"]
NS = [1, 2, 5, 9, 20]

with open("results_C.csv", "w", newline="") as f:
    w = csv.writer(f)
    header = BASE_COLS[:]
    for N in NS:
        header.append(f"eps0_N{N}")
    for N in NS:
        header.append(f"eps1_N{N}")
    w.writerow(header)
    for k, v in R.items():
        row = []
        for col in BASE_COLS:
            val = v.get(col)
            if isinstance(val, list):
                val = ";".join(str(x) for x in val)
            row.append(val)
        for N in NS:
            row.append(v["eps0"][str(N)] if v.get("eps0") else None)
        for N in NS:
            row.append(v["eps1"][str(N)] if v.get("eps1") else None)
        w.writerow(row)

print("wrote results_C.csv")
