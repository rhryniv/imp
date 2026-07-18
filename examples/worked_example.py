"""Reproduce (approximately) Table 1 of the paper's worked example:
J0 = [pi/6, pi/4], J1 = [pi/2, pi], for n = 3,5,7,9 and mu0 = 0.5,1.0,1.5.

Run from the repository root:  python3 examples/worked_example.py
"""
import numpy as np

from scattering.design_full import design_filter_full, verify_design
from scattering.polish import polish_design

J0 = [(np.pi / 6, np.pi / 4)]
J1 = [(np.pi / 2, np.pi)]

TABLE1 = {
    (3, 0.5): 1.6e-1, (3, 1.0): 8.4e-1, (3, 1.5): 2.8e0,
    (5, 0.5): 7.1e-3, (5, 1.0): 3.6e-2, (5, 1.5): 1.2e-1,
    (7, 0.5): 2.6e-4, (7, 1.0): 1.3e-3, (7, 1.5): 4.3e-3,
    (9, 0.5): 1.0e-5, (9, 1.0): 6.1e-5, (9, 1.5): 2.0e-4,
}


def main():
    print(f"{'n':>3} {'mu0':>5} {'delta1 (ours)':>14} {'delta1 (Table 1)':>17} {'ratio':>8} {'verified':>9}")
    for n in (3, 5, 7, 9):
        for mu0 in (0.5, 1.0, 1.5):
            res = design_filter_full(n, J0, J1, mu0)
            a_pol, u_pol, _ = polish_design(res.a, J0, J1, mu0, res.sigma)
            delta1 = np.sqrt(max(u_pol, 0.0)) - 1.0
            verified = verify_design(a_pol, J0, J1, mu0, res.sigma, u_pol, tol=1e-6)
            target = TABLE1[(n, mu0)]
            print(f"{n:3d} {mu0:5.1f} {delta1:14.4e} {target:17.4e} {delta1 / target:8.2f} {str(verified):>9}")


if __name__ == "__main__":
    main()
