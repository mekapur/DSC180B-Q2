import numpy as np
import pandas as pd
import pytest

from src.eval.compare import (
    categorical_accuracy,
    jaccard_similarity,
    relative_error,
    spearman_rho,
    total_variation_distance,
)


class TestRelativeError:
    def test_equal_values(self):
        assert relative_error(5.0, 5.0) == 0.0

    def test_both_zero(self):
        assert relative_error(0.0, 0.0) == 0.0

    def test_real_zero_synth_nonzero(self):
        assert relative_error(0.0, 1.0) == float("inf")

    def test_positive_values(self):
        assert abs(relative_error(100.0, 110.0) - 0.1) < 1e-10

    def test_symmetric_in_absolute(self):
        re1 = relative_error(100.0, 80.0)
        re2 = relative_error(100.0, 120.0)
        assert abs(re1 - re2) < 1e-10


class TestTotalVariation:
    def test_identical_distributions(self):
        p = np.array([0.5, 0.3, 0.2])
        assert total_variation_distance(p, p) == 0.0

    def test_disjoint_distributions(self):
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert abs(total_variation_distance(p, q) - 1.0) < 1e-10

    def test_unnormalized_inputs(self):
        p = np.array([50, 30, 20])
        q = np.array([40, 35, 25])
        tv = total_variation_distance(p, q)
        assert 0 <= tv <= 1

    def test_both_zero(self):
        assert total_variation_distance(np.array([0, 0]), np.array([0, 0])) == 0.0


class TestSpearmanRho:
    def test_perfect_correlation(self):
        r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(spearman_rho(r, r) - 1.0) < 1e-10

    def test_perfect_anticorrelation(self):
        r = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert abs(spearman_rho(r, s) - (-1.0)) < 1e-10

    def test_single_element(self):
        assert spearman_rho(np.array([1.0]), np.array([2.0])) == 0.0

    def test_constant_values(self):
        r = np.array([1.0, 1.0, 1.0])
        s = np.array([1.0, 2.0, 3.0])
        assert spearman_rho(r, s) == 0.0


class TestJaccard:
    def test_identical_sets(self):
        assert jaccard_similarity({1, 2, 3}, {1, 2, 3}) == 1.0

    def test_disjoint_sets(self):
        assert jaccard_similarity({1, 2}, {3, 4}) == 0.0

    def test_partial_overlap(self):
        assert abs(jaccard_similarity({1, 2, 3}, {2, 3, 4}) - 0.5) < 1e-10

    def test_both_empty(self):
        assert jaccard_similarity(set(), set()) == 1.0


class TestCategoricalAccuracy:
    def test_perfect_match(self):
        r = pd.Series(["chrome", "edge"], index=["US", "DE"])
        assert categorical_accuracy(r, r) == 1.0

    def test_no_match(self):
        r = pd.Series(["chrome", "edge"], index=["US", "DE"])
        s = pd.Series(["edge", "chrome"], index=["US", "DE"])
        assert categorical_accuracy(r, s) == 0.0

    def test_partial_overlap_keys(self):
        r = pd.Series(["chrome", "edge", "firefox"], index=["US", "DE", "JP"])
        s = pd.Series(["chrome", "edge"], index=["US", "DE"])
        acc = categorical_accuracy(r, s)
        assert abs(acc - 2.0 / 3.0) < 1e-10
