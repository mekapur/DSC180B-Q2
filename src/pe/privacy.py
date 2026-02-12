"""Privacy accounting for the Gaussian mechanism in Private Evolution.

Implements the analytic Gaussian mechanism from Balle and Wang (2018),
which provides a tight closed-form (epsilon, delta)-DP guarantee for
Gaussian noise addition. For T iterations of PE, the effective sensitivity
scales as sqrt(T) via adaptive composition (Dong et al. 2022, Corollary 3.3).
"""

from scipy.stats import norm
from scipy.optimize import brentq


def analytic_gaussian_delta(sigma: float, epsilon: float, sensitivity: float = 1.0) -> float:
    """Compute delta for the analytic Gaussian mechanism (Balle and Wang, 2018).

    Given noise scale sigma and privacy parameter epsilon, returns the
    tightest delta such that adding N(0, sigma^2) noise to a query with
    the given sensitivity satisfies (epsilon, delta)-DP.
    """
    import math
    mu = sensitivity / sigma
    term1 = norm.cdf(mu / 2 - epsilon / mu)
    term2_log = epsilon + norm.logcdf(-mu / 2 - epsilon / mu)
    if term2_log > 700:
        return float("inf")
    term2 = math.exp(term2_log)
    return term1 - term2


def calibrate_sigma(epsilon: float, delta: float, T: int, sensitivity: float = 1.0) -> float:
    """Find the minimum sigma satisfying (epsilon, delta)-DP for T iterations.

    Uses bisection (brentq) to invert the analytic Gaussian delta formula.
    The effective sensitivity is sensitivity * sqrt(T), reflecting adaptive
    composition of T Gaussian mechanism applications.
    """
    effective_sensitivity = sensitivity * (T ** 0.5)

    def objective(sigma):
        return analytic_gaussian_delta(sigma, epsilon, effective_sensitivity) - delta

    lo, hi = 0.01, 10000.0
    while objective(hi) > 0:
        hi *= 2
    return brentq(objective, lo, hi, xtol=1e-8)


def compute_epsilon(sigma: float, delta: float, T: int, sensitivity: float = 1.0) -> float:
    """Compute the achieved epsilon for a given sigma, delta, and T.

    Inverse of calibrate_sigma: given noise scale and delta, finds the
    tightest epsilon. Used for post-hoc privacy reporting.
    """
    effective_sensitivity = sensitivity * (T ** 0.5)

    def objective(eps):
        return analytic_gaussian_delta(sigma, eps, effective_sensitivity) - delta

    lo, hi = 0.0, 1000.0
    try:
        return brentq(objective, lo, hi, xtol=1e-8)
    except ValueError:
        return float("inf")
