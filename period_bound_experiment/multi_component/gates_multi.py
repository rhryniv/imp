"""Pre-sweep regression gate (spec Sec 4): re-run the Sec 6.2 geometry
(I1=[0,pi/4], I0=[5pi/6,pi], single component each) at n=3,d=1,sigma=+1
and confirm delta=4.6722e-4 to 5 s.f. Abort the sweep on failure.
"""
from __future__ import annotations

import numpy as np

from optimize_multi import Problem, solve_case, GEOMETRIES

RNG_SEED = 20260827

# Register the Sec 6.2 single-component geometry under the same registry
# so Problem(...)/solve_case(...) work unchanged for the gate.
GEOMETRIES["G62"] = {
    "I1": [(0.0, np.pi / 4)],
    "I0": [(5 * np.pi / 6, np.pi)],
    "pi_in": "I0",
    "pi_component_index": 0,
}


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
