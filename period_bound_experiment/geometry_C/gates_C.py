"""Pre-sweep regression gate: Sec 6.2 geometry, n=3,d=1,sigma=+1,
delta=4.6722e-4 to 5 s.f. Abort the sweep on failure.
"""
from __future__ import annotations

import numpy as np

from optimize_C import solve_case

RNG_SEED = 20260828


def run_gate():
    rng = np.random.default_rng(RNG_SEED)
    best, nconv, ntot, wall = solve_case("G62", 3, 1, (1,), rng, n_random=200, n_antisym=50)
    delta = best["delta_fine"] if best else None
    ok = best is not None and np.isclose(delta, 4.6722e-4, rtol=0, atol=5e-8)
    print(f"Gate (Sec 6.2, n=3,d=1,sigma=+1): delta={delta} (target 4.6722e-4)  "
          f"nconv={nconv}/{ntot}  wall={wall:.1f}s  -> {'PASS' if ok else 'FAIL'}")
    return ok, {"delta": delta, "target": 4.6722e-4, "nconv": nconv, "ntot": ntot, "wall_s": wall}


if __name__ == "__main__":
    ok, res = run_gate()
    if not ok:
        raise SystemExit("GATE FAILED -- aborting per spec Sec 4.")
