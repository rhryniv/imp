"""Single entry point: gates (abort on failure) -> full 66-case sweep
(checkpointed, resumable) -> results.csv. Deterministic given the RNG
seeds in config.json / gates.py / run_sweep.py.

    python3 run.py

Requires: numpy, scipy (see config.json for exact versions used). No
network access. Modules: core.py (forward model), optimize.py (SLSQP
problem + starts), gates.py, run_sweep.py, make_outputs.py.
"""
from __future__ import annotations

import sys

from gates import run_gates


def main():
    print("=== Regression gates (spec Sec 4) ===")
    ok, _ = run_gates()
    if not ok:
        print("\nGATES FAILED -- aborting per spec Sec 4 instruction "
              "('stop and report the discrepancy rather than proceeding').")
        sys.exit(1)

    print("\n=== Full 66-case sweep ===")
    import run_sweep
    run_sweep.main()

    print("\n=== Building results.csv ===")
    import make_outputs  # noqa: F401 (module runs its work at import time)


if __name__ == "__main__":
    main()
