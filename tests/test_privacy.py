import math

import pytest

from src.pe.privacy import analytic_gaussian_delta, calibrate_sigma, compute_epsilon


class TestAnalyticGaussianDelta:
    def test_large_sigma_small_delta(self):
        delta = analytic_gaussian_delta(sigma=10.0, epsilon=1.0)
        assert delta < 0.01

    def test_small_sigma_large_delta(self):
        delta = analytic_gaussian_delta(sigma=0.1, epsilon=1.0)
        assert delta > 0.1

    def test_zero_epsilon_positive_delta(self):
        delta = analytic_gaussian_delta(sigma=1.0, epsilon=0.0)
        assert delta >= 0

    def test_monotone_in_sigma(self):
        d1 = analytic_gaussian_delta(sigma=1.0, epsilon=1.0)
        d2 = analytic_gaussian_delta(sigma=2.0, epsilon=1.0)
        assert d1 > d2

    def test_monotone_in_epsilon(self):
        d1 = analytic_gaussian_delta(sigma=1.0, epsilon=0.5)
        d2 = analytic_gaussian_delta(sigma=1.0, epsilon=1.0)
        assert d1 > d2


class TestCalibrateSigma:
    def test_known_value_T1(self):
        sigma = calibrate_sigma(epsilon=4.0, delta=1e-5, T=1)
        assert 1.0 < sigma < 1.2

    def test_larger_epsilon_smaller_sigma(self):
        s1 = calibrate_sigma(epsilon=1.0, delta=1e-5, T=1)
        s2 = calibrate_sigma(epsilon=4.0, delta=1e-5, T=1)
        assert s1 > s2

    def test_more_iterations_larger_sigma(self):
        s1 = calibrate_sigma(epsilon=4.0, delta=1e-5, T=1)
        s2 = calibrate_sigma(epsilon=4.0, delta=1e-5, T=10)
        assert s2 > s1

    def test_round_trip_with_compute_epsilon(self):
        eps_target = 4.0
        delta = 1e-5
        T = 1
        sigma = calibrate_sigma(eps_target, delta, T)
        eps_actual = compute_epsilon(sigma, delta, T)
        assert abs(eps_target - eps_actual) < 1e-4


class TestComputeEpsilon:
    def test_large_sigma_small_epsilon(self):
        eps = compute_epsilon(sigma=100.0, delta=1e-5, T=1)
        assert eps < 1.0

    def test_small_sigma_large_epsilon(self):
        eps = compute_epsilon(sigma=0.5, delta=1e-5, T=1)
        assert eps > 1.0

    def test_monotone_in_sigma(self):
        e1 = compute_epsilon(sigma=1.0, delta=1e-5, T=1)
        e2 = compute_epsilon(sigma=2.0, delta=1e-5, T=1)
        assert e1 > e2
