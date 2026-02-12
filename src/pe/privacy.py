from scipy.stats import norm
from scipy.optimize import brentq


def analytic_gaussian_delta(sigma: float, epsilon: float, sensitivity: float = 1.0) -> float:
    import math
    mu = sensitivity / sigma
    term1 = norm.cdf(mu / 2 - epsilon / mu)
    term2_log = epsilon + norm.logcdf(-mu / 2 - epsilon / mu)
    if term2_log > 700:
        return float("inf")
    term2 = math.exp(term2_log)
    return term1 - term2


def calibrate_sigma(epsilon: float, delta: float, T: int, sensitivity: float = 1.0) -> float:
    effective_sensitivity = sensitivity * (T ** 0.5)

    def objective(sigma):
        return analytic_gaussian_delta(sigma, epsilon, effective_sensitivity) - delta

    lo, hi = 0.01, 10000.0
    while objective(hi) > 0:
        hi *= 2
    return brentq(objective, lo, hi, xtol=1e-8)


def compute_epsilon(sigma: float, delta: float, T: int, sensitivity: float = 1.0) -> float:
    effective_sensitivity = sensitivity * (T ** 0.5)

    def objective(eps):
        return analytic_gaussian_delta(sigma, eps, effective_sensitivity) - delta

    lo, hi = 0.0, 1000.0
    try:
        return brentq(objective, lo, hi, xtol=1e-8)
    except ValueError:
        return float("inf")
