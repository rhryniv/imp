"""E4: phase variation V_J(psi), J=I_1, n=1..9, both specs. Uses the best
feasible p1 from E2 (n=1..6, min delta over all L,sigma), falling back to
E3's construction for odd n outside E2's range, and to a direct
Qhat_n-construction + spectral-factor (no layer-strip needed for this
purpose -- only p1 itself matters) for whatever n neither covers.
psi = continuous branch of arg p1(e^{-i theta}) on I_1, via fine grid +
np.angle + np.unwrap (fine grid = 20000 points, per hazard #5's fine-grid
standard used elsewhere in this brief).
"""
from __future__ import annotations

import json

import numpy as np

from geometry import setup
from e2_core import forward_recursion_np, poly_eval_np
import run_e3

N_LIST = list(range(1, 10))
GRID_N = 20000


def V_J(alphas, t, grid_n=GRID_N):
    p1, _ = forward_recursion_np(np.asarray(alphas, dtype=float))
    thetas = np.linspace(0.0, t, grid_n)
    w = np.exp(-1j * thetas)
    p1w = poly_eval_np(p1, w)
    psi = np.unwrap(np.angle(p1w))
    return float(np.sum(np.abs(np.diff(psi))))


def qhat_fallback_alphas(n, geo):
    q, dn = run_e3.qhat_n_cheb_coeffs(n, geo)
    f = run_e3.cosine_coeffs(q)
    p1, outside, rho, max_imag = run_e3.spectral_factor(f, run_e3.DPS)
    p2, touches = run_e3.build_p2(n, geo, dn)
    alphas, diag = run_e3.downward_peel(p1, p2, run_e3.DPS)
    return np.array([float(a) for a in alphas])


def main():
    with open("../data/e2_results.json") as f:
        e2 = json.load(f)
    with open("../data/e3_results.json") as f:
        e3 = json.load(f)

    results = {}
    for name in ["A", "B"]:
        geo = setup(name, 30)
        t = float(geo["t"])
        results[name] = {}
        print(f"\n=== Spec {name} ===")
        for n in N_LIST:
            alphas = None
            source = None
            n_key = str(n)
            if n_key in e2.get(name, {}):
                best_delta = None
                best_alphas = None
                for cellkey, cell in e2[name][n_key].items():
                    if cell.get("feasible") and cell.get("delta_polished") is not None:
                        if best_delta is None or cell["delta_polished"] < best_delta:
                            best_delta = cell["delta_polished"]
                            best_alphas = cell["alphas"]
                if best_alphas is not None:
                    alphas = np.array(best_alphas)
                    source = f"E2 best (delta={best_delta:.4e})"
            if alphas is None and n_key in e3.get(name, {}):
                # E3's stored alphas were produced with a DPS=60 mpmath
                # pipeline but written out as python floats already
                r = e3[name][n_key]
                if "alphas" in r:
                    alphas = np.array(r["alphas"])
                    source = "E3 construction"
            if alphas is None:
                geo50 = setup(name, 50)
                try:
                    alphas = qhat_fallback_alphas(n, geo50)
                    source = "Qhat_n+spectral-factor fallback (no E2/E3 coverage)"
                except Exception as exc:
                    results[name][n] = {"error": str(exc)}
                    print(f"  n={n}  FAILED: {exc}")
                    continue

            vj = V_J(alphas, t)
            cap = 2 * (np.pi + vj) / (t)  # prop:sweep: n < 2*(pi+V_J)/|J|, |J|=t here (I_1=[0,t])
            results[name][n] = {"V_J": vj, "V_J_over_pi": vj / np.pi, "source": source,
                                 "sweep_cap_n": cap}
            print(f"  n={n}  V_J={vj:.6f} ({vj/np.pi:.4f} pi)  source=[{source}]  sweep_cap(n<)={cap:.3f}")

    with open("../data/e4_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nsaved data/e4_results.json")
    return results


if __name__ == "__main__":
    main()
