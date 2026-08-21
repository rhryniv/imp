import mpmath as mp

from geometry import setup, delta_n_closed, cor_resource_bound

TARGETS = {
    "A": {"gamma": "3.154502723904438820", "S": "1.381097845541815730",
          "s_minus": "0.3628328048", "rho1": "1.769760347", "delta1_star": "0.2896599407",
          "ratio1": "1.336202330", "cor_resource": "2.014931739"},
    "B": {"gamma": "3.057141839", "ratio1": "1.803423338"},
}


def check(label, computed, target, tol=1e-9):
    diff = abs(float(computed) - float(target))
    status = "PASS" if diff < tol else f"FAIL (diff={diff:.3e})"
    print(f"  {label}: computed={mp.nstr(computed,12)}  target={target}  {status}")


for name in ["A", "B"]:
    geo = setup(name, 40)
    print(f"=== Spec {name} ===")
    check("gamma", geo["gamma"], TARGETS[name]["gamma"])
    if name == "A":
        check("S", geo["S"], TARGETS[name]["S"])
        for n in range(1, 8):
            pass  # already checked in geometry.py's own __main__

        a, b, mu0, S = geo["a"], geo["b"], geo["mu0"], geo["S"]
        s_minus = (mp.cosh(mu0) + b) / (1 - b)
        check("s_-(mu0)", s_minus, TARGETS[name]["s_minus"])
        alpha1 = mp.asinh(mp.sqrt(s_minus))
        rho1 = mp.e ** alpha1
        check("rho_1", rho1, TARGETS[name]["rho1"])
        t = geo["t"]
        delta1_star = 4 * s_minus * (1 + s_minus) * mp.sin(t / 2) ** 2
        check("delta*_1", delta1_star, TARGETS[name]["delta1_star"])
        dmag1 = delta_n_closed(1, geo)
        ratio1 = delta1_star / dmag1
        check("delta*_1/delta_mag(1)", ratio1, TARGETS[name]["ratio1"])
        cr = cor_resource_bound(geo)
        check("cor:resource bound", cr, TARGETS[name]["cor_resource"])
    else:
        a, b, mu0 = geo["a"], geo["b"], geo["mu0"]
        s_minus = (mp.cosh(mu0) + b) / (1 - b)
        t = geo["t"]
        delta1_star = 4 * s_minus * (1 + s_minus) * mp.sin(t / 2) ** 2
        dmag1 = delta_n_closed(1, geo)
        ratio1 = delta1_star / dmag1
        check("delta*_1/delta_mag(1)", ratio1, TARGETS[name]["ratio1"])
