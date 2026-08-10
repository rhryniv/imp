"""Forward problem: alpha's -> a -> kappa_B -> T_N.

alphas = (alpha_0, ..., alpha_n) are the log-impedance contrasts of an
n-layer block (eq. 3.1 in the paper). Everything here is pure evaluation:
no optimization, no factorization (that's inverse.py and sdp_design.py).
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as C


def forward_reconstruct(alphas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build (p1, p2) from alphas via the recursion of eqs. 4.5-4.6:

        p_1^(0) = cosh(alpha_0),   p_2^(0) = sinh(alpha_0)
        p_1^(j) = cosh(alpha_j) p_1^(j-1)     + sinh(alpha_j) [w p_2^(j-1)]
        p_2^(j) = sinh(alpha_j) p_1^(j-1)     + cosh(alpha_j) [w p_2^(j-1)]

    p1, p2 are returned as coefficient arrays of increasing powers of w,
    length n+1 (zero-padded to a common length along the way). This is
    the forward transfer-matrix recursion -- always well defined for any
    real alphas, and it automatically satisfies |p1|^2-|p2|^2=1 on |w|=1
    and p1 zero-free in the closed unit disc (Prop 4.4), i.e. it can only
    ever produce a *realizable* q~_1.
    """
    alphas = np.asarray(alphas, dtype=float)
    P1, P2 = np.array([np.cosh(alphas[0])]), np.array([np.sinh(alphas[0])])
    for alpha_j in alphas[1:]:
        ch, sh = np.cosh(alpha_j), np.sinh(alpha_j)
        P1_padded = np.concatenate([P1, [0.0]])         # p_1^(j-1), degree bumped (no shift)
        wP2 = np.concatenate([[0.0], P2])                # w * p_2^(j-1)  (shift up by one)
        P1_new = ch * P1_padded + sh * wP2
        P2_new = sh * P1_padded + ch * wP2
        P1, P2 = P1_new, P2_new
    return P1, P2


def a_from_alphas(alphas: np.ndarray) -> np.ndarray:
    """alphas -> a = q~_1 coefficients. a_m = p1[n-m] (eq. after 5.6: a_m = c_{n-m})."""
    p1, _ = forward_reconstruct(alphas)
    return p1[::-1].copy()


def eval_poly(coeffs: np.ndarray, w: np.ndarray) -> np.ndarray:
    """p(w) = sum_k coeffs[k] w^k, coeffs in increasing powers (Horner)."""
    coeffs = np.asarray(coeffs)
    w = np.asarray(w)
    result = np.zeros_like(w, dtype=np.result_type(coeffs.dtype, w.dtype, complex))
    for c in coeffs[::-1]:
        result = result * w + c
    return result


def forward_matrix_product(alphas: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(p1, p2) evaluated at theta via the LITERAL per-layer SU(1,1) transfer
    matrices of Sec. 3.1/3.2 (M(alpha_j, k*x_j), x_j=j*h), rather than
    forward_reconstruct's coefficient-vector recursion -- an independently
    coded second path for Stage 1's cross-check (spec Sec. 2: "(a) the
    recursion above; (b) the direct SU(1,1) product of Sec. 3's transfer
    matrices... must agree to machine precision").

    Uses Sec. 4.1's own identification p_{1,j}(w) = q_{1,j}(k) exactly, and
    p_{2,j}(w) = e^{-2i k x_j} q_{2,j}(k) (eq. after 4.2): tracks the
    *physical* partial transfer-matrix entries q_{1,j}(k), q_{2,j}(k) via

        q_{1,j} = cosh(a_j) q_{1,j-1} + sinh(a_j) e^{-2ikx_j} q_{2,j-1}
        q_{2,j} = sinh(a_j) e^{+2ikx_j} q_{1,j-1} + cosh(a_j) q_{2,j-1}

    with explicit *position-dependent* phases e^{+-2ikx_j} at every layer
    (x_j=j*h, so k*x_j = j*theta/2, i.e. e^{-2ikx_j}=e^{-ij*theta}) --
    algebraically equivalent to path (a)'s single running w-multiplication
    per step, but via a genuinely different intermediate computation (no
    coefficient vectors, no padding/shifting, pure pointwise complex
    arithmetic throughout), then converts back to (p1, p2) via the single
    w^n phase correction on q2 at the very end, per the identification
    above.

    Verified by hand against fixture F1 (n=1, alpha=(log2,-log2)): matches
    the manuscript's own closed form p1(w) = 25/16 - (9/16)w exactly.
    """
    alphas = np.asarray(alphas, dtype=float)
    n = len(alphas) - 1
    theta = np.atleast_1d(np.asarray(theta, dtype=float))

    q1 = np.full(theta.shape, np.cosh(alphas[0]), dtype=complex)   # q_{1,0}(k) = cosh(alpha_0)
    q2 = np.full(theta.shape, np.sinh(alphas[0]), dtype=complex)   # q_{2,0}(k) = sinh(alpha_0)
    for j in range(1, n + 1):
        ch, sh = np.cosh(alphas[j]), np.sinh(alphas[j])
        phase = np.exp(-1j * j * theta)          # e^{-2ikx_j}, x_j=j*h, k*x_j=j*theta/2
        q1, q2 = ch * q1 + sh * phase * q2, sh * np.conj(phase) * q1 + ch * q2

    p1 = q1
    p2 = np.exp(-1j * n * theta) * q2            # w^n = e^{-in*theta}, phase correction at j=n
    return p1, p2


def forward_with_grad(alphas: np.ndarray, theta: np.ndarray):
    """Pointwise evaluation of p1(theta), Q(theta)=|p1|^2, kappa(theta), and
    their exact gradients w.r.t. every alpha_j, at given theta nodes
    (manuscript Sec. 6.3, "Exact gradients"). Unlike forward_reconstruct
    (which returns p1/p2 as *coefficient arrays*, used for the a-vector /
    Chebyshev-domain machinery), this evaluates directly at theta via
    w=exp(-i*theta), matching the direct-optimisation grid-constraint use
    case (sdp_design.design_direct) where per-node analytic gradients are
    needed for SLSQP.

    Recursion: p^{(j)} = A_j p^{(j-1)}, p^{(-1)}=(1,0)^T, with
        A_j = C(alpha_j) diag(1,w),  C(alpha)=exp(alpha K),  K=[[0,1],[1,0]],
    so that d/dalpha_j A_j = K A_j exactly. With prefix products
    S_j = A_j...A_0 and suffix products L_j = A_n...A_{j+1}, the product
    rule gives d/dalpha_j (A_n...A_0) = L_j K S_j for every j -- one
    forward sweep (caching p^{(j)} = S_j (1,0)^T) and one backward sweep
    (caching L_j) give all n+1 derivatives at every node in O(n) work per
    node, O(1) extra work per (node, j) pair thereafter.

    Returns a dict with p1, Q, kappa (each shape (M,)) and dp1, dQ, dkappa
    (each shape (n+1, M), one row per alpha_j, *before* the free-variable
    elimination alpha_n=-sum(alpha_{<n}) -- see grad_free_vars below for
    that last step).
    """
    alphas = np.asarray(alphas, dtype=float)
    n = len(alphas) - 1
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    M = theta.shape[0]
    w = np.exp(-1j * theta)                     # (M,)
    wbar_n = np.exp(1j * n * theta)              # conj(w)^n, |w|=1 on the real theta axis

    ch, sh = np.cosh(alphas), np.sinh(alphas)    # (n+1,)

    # A_j, shape (n+1, M, 2, 2)
    A = np.empty((n + 1, M, 2, 2), dtype=complex)
    A[:, :, 0, 0] = ch[:, None]
    A[:, :, 0, 1] = sh[:, None] * w[None, :]
    A[:, :, 1, 0] = sh[:, None]
    A[:, :, 1, 1] = ch[:, None] * w[None, :]

    # Forward sweep: p[j+1] = p^{(j)} for j=-1,...,n (index shifted by 1; p[0]=p^{(-1)})
    p = np.empty((n + 2, M, 2), dtype=complex)
    p[0, :, 0], p[0, :, 1] = 1.0, 0.0
    for j in range(n + 1):
        p[j + 1, :, 0] = A[j, :, 0, 0] * p[j, :, 0] + A[j, :, 0, 1] * p[j, :, 1]
        p[j + 1, :, 1] = A[j, :, 1, 0] * p[j, :, 0] + A[j, :, 1, 1] * p[j, :, 1]
    p1, p2 = p[n + 1, :, 0], p[n + 1, :, 1]

    # Backward sweep: L[j] = L_j = A_n...A_{j+1}, for j=0,...,n (L_n = I)
    L = np.empty((n + 1, M, 2, 2), dtype=complex)
    Lcur = np.zeros((M, 2, 2), dtype=complex)
    Lcur[:, 0, 0] = Lcur[:, 1, 1] = 1.0
    L[n] = Lcur
    for j in range(n, 0, -1):
        Lcur = np.einsum('mij,mjk->mik', Lcur, A[j])
        L[j - 1] = Lcur

    # d/dalpha_j p^{(n)} = L_j K p^{(j)}, K=[[0,1],[1,0]] i.e. swaps components
    Kp0, Kp1 = p[1:, :, 1], p[1:, :, 0]          # (n+1, M): K @ p^{(j)} for j=0..n
    dp1 = L[:, :, 0, 0] * Kp0 + L[:, :, 0, 1] * Kp1
    dp2 = L[:, :, 1, 0] * Kp0 + L[:, :, 1, 1] * Kp1

    Q = np.abs(p1) ** 2
    dQ = 2.0 * np.real(np.conj(p1)[None, :] * dp1)
    kappa = np.real(wbar_n * p1)
    dkappa = np.real(wbar_n[None, :] * dp1)

    return {
        "p1": p1, "p2": p2, "Q": Q, "kappa": kappa,
        "dp1": dp1, "dp2": dp2, "dQ": dQ, "dkappa": dkappa,
    }


def grad_free_vars(dvals: np.ndarray) -> np.ndarray:
    """Chain-rule step for the free-variable elimination alpha_n :=
    -sum_{j<n} alpha_j (manuscript Sec. 6.3): given dvals of shape
    (n+1, M) -- one row per alpha_j, from forward_with_grad -- returns the
    gradient w.r.t. the n free variables alpha_0,...,alpha_{n-1}, shape
    (n, M): d/dalpha_j(free) = d/dalpha_j - d/dalpha_n for j<n."""
    return dvals[:-1] - dvals[-1]


def kappa_B(a: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """kappa_B(theta) = Re(sum_m a_m e^{i m theta}) = sum_m a_m cos(m theta)
    = Chebyshev evaluation of a at x = cos(theta) (eq. 5.7-5.8)."""
    return C.chebval(np.cos(theta), a)


def autocorr(a: np.ndarray) -> np.ndarray:
    """f_0,...,f_n with Q(theta) = f_0 + 2*sum_{m>=1} f_m cos(m*theta),
    f_m = sum_l a_l a_{l+m} (eq. f-autocorr). Reversal-invariant (the
    autocorrelation of c and of its reverse a, a_m=c_{n-m}, coincide), so
    it makes no difference which coefficient vector is passed in."""
    a = np.asarray(a, dtype=float)
    n = len(a) - 1
    return np.array([np.sum(a[: n + 1 - m] * a[m:]) for m in range(n + 1)])


def Q_from_autocorr(a: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Q(theta) = |q~_1(theta)|^2, computed as a single REAL cosine sum via
    the autocorrelation f=autocorr(a) -- f_0 + 2*sum_{m>=1} f_m cos(m theta)
    -- rather than evaluating the complex sum q~_1(theta)=sum a_m e^{im
    theta} and squaring its modulus.

    This matters numerically: the a_m span a wide dynamic range (products
    of cosh/sinh(alpha_j) compounding through the layer recursion), so the
    complex sum can suffer heavy cancellation even though |q~_1|>=1 always
    (constraint A) -- and squaring an already-cancelled complex value
    roughly doubles its relative error on top. Replacing "complex sum, then
    square" with a single bounded real sum (the same pattern kappa_B
    already uses via chebval) removes both the complex arithmetic and the
    squaring step from the hot path. f itself is a one-off O(n^2)
    computation per design, not repeated per grid point, so any
    cancellation there is paid once rather than amplified across the whole
    grid.
    """
    f = autocorr(a)
    scale = np.concatenate([[1.0], 2.0 * np.ones(len(f) - 1)])
    return C.chebval(np.cos(theta), scale * f)


def q1_abs_sq(a: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """|q~_1(theta)|^2 = G(theta), via the numerically stable autocorrelation
    route (Q_from_autocorr) rather than evaluating the complex sum
    q~_1(theta)=sum a_m e^{im theta} and squaring its modulus."""
    return Q_from_autocorr(a, theta)


def q2_abs_sq(a: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """|q_2(theta)|^2 = |q~_1(theta)|^2 - 1 (eq. 5.11), clipped to >= 0."""
    return np.clip(q1_abs_sq(a, theta) - 1.0, 0.0, None)


def _chebyshev_U(m: int, x: np.ndarray) -> np.ndarray:
    """Chebyshev polynomial of the second kind, U_m(x)."""
    x = np.asarray(x, dtype=float)
    if m < 0:
        return np.zeros_like(x)
    U_km1 = np.ones_like(x)          # U_0
    if m == 0:
        return U_km1
    U_k = 2 * x                       # U_1
    for _ in range(2, m + 1):
        U_km1, U_k = U_k, 2 * x * U_k - U_km1
    return U_k


def transmission_TN(a: np.ndarray, theta: np.ndarray, N: int) -> np.ndarray:
    """T_N(theta) = 1 / (1 + |q_2|^2 * U_{N-1}(kappa_B)^2)   (eq. 4.13)."""
    kap = kappa_B(a, theta)
    q2sq = q2_abs_sq(a, theta)
    U = _chebyshev_U(N - 1, kap)
    return 1.0 / (1.0 + q2sq * U ** 2)


def plot_filter(a, N_values, theta_range=(1e-3, np.pi - 1e-3), I0=None, I1=None,
                 n_grid=4000, savepath=None):
    """Figure with panels for kappa_B(theta), |q~_1(theta)|^2, and T_N(theta)
    (one curve per N in N_values). Shades I0 (stop, red) / I1 (pass, green)
    if given (each a list of (lo, hi) pairs in theta).

    Returns the matplotlib Figure; also saves to savepath if given.
    """
    import matplotlib.pyplot as plt

    theta = np.linspace(theta_range[0], theta_range[1], n_grid)
    kap = kappa_B(a, theta)
    G = q1_abs_sq(a, theta)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].plot(theta, kap, color="black")
    axes[0].axhline(1.0, color="gray", lw=0.7, ls="--")
    axes[0].axhline(-1.0, color="gray", lw=0.7, ls="--")
    axes[0].set_ylabel(r"$\kappa_B(\theta)$")

    axes[1].plot(theta, G, color="black")
    axes[1].axhline(1.0, color="gray", lw=0.7, ls="--")
    axes[1].set_ylabel(r"$|\tilde q_1(\theta)|^2$")

    for N in N_values:
        axes[2].plot(theta, transmission_TN(a, theta, N), label=f"N={N}")
    axes[2].set_ylabel(r"$T_N(\theta)$")
    axes[2].set_xlabel(r"$\theta = 2kh$")
    axes[2].legend()

    for ax in axes:
        for lo, hi in (I0 or []):
            ax.axvspan(lo, hi, color="red", alpha=0.15)
        for lo, hi in (I1 or []):
            ax.axvspan(lo, hi, color="green", alpha=0.15)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig
