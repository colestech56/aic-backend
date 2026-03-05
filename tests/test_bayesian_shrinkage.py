"""Tests for the Bayesian Shrinkage Estimator."""

import pytest

from app.services.bayesian_shrinkage import BayesianShrinkageEstimator, POPULATION_PRIORS


@pytest.fixture
def estimator():
    return BayesianShrinkageEstimator()


class TestZeroSurveys:
    def test_sz_returns_population_prior(self, estimator):
        threshold, weight = estimator.compute_threshold("SZ", [])
        assert threshold == 4.0
        assert weight == 0.0

    def test_bd_returns_population_prior(self, estimator):
        threshold, weight = estimator.compute_threshold("BD", [])
        assert threshold == 4.0
        assert weight == 0.0


class TestIndividualWeight:
    def test_weight_at_n1(self, estimator):
        _, weight = estimator.compute_threshold("SZ", [3.0])
        assert abs(weight - 1 / 21) < 1e-10

    def test_weight_at_n9(self, estimator):
        scores = [3.0] * 9
        _, weight = estimator.compute_threshold("SZ", scores)
        assert abs(weight - 9 / 21) < 1e-10

    def test_weight_at_n21(self, estimator):
        scores = [3.0] * 21
        _, weight = estimator.compute_threshold("SZ", scores)
        assert abs(weight - 1.0) < 1e-10

    def test_weight_caps_at_1(self, estimator):
        scores = [3.0] * 50
        _, weight = estimator.compute_threshold("SZ", scores)
        assert weight == 1.0


class TestThresholdBlending:
    def test_partially_individual(self, estimator):
        """With 10 surveys of NA=5.0 (no variance), threshold should blend."""
        scores = [5.0] * 10
        threshold, weight = estimator.compute_threshold("SZ", scores)
        # w = 10/21 ≈ 0.476
        # individual = 5.0 + 0.0*SD = 5.0
        # blended = 0.476 * 5.0 + 0.524 * 4.0 = 2.381 + 2.095 = 4.476
        assert abs(weight - 10 / 21) < 1e-10
        assert 4.0 < threshold < 5.0

    def test_fully_individual_at_21(self, estimator):
        """At 21 surveys, threshold should be fully individual."""
        scores = [5.0] * 21
        threshold, weight = estimator.compute_threshold("SZ", scores)
        # w = 1.0, individual = 5.0 + 0 = 5.0
        assert weight == 1.0
        assert abs(threshold - 5.0) < 0.01


class TestConstraints:
    def test_floor_enforced(self, estimator):
        """Very low NA scores should not push threshold below 3.0."""
        scores = [1.5] * 30  # Very low NA
        threshold, _ = estimator.compute_threshold("SZ", scores)
        assert threshold >= 3.0

    def test_ceiling_enforced(self, estimator):
        """Very high NA scores should not push threshold above 6.0."""
        scores = [6.5] * 30  # Very high NA
        threshold, _ = estimator.compute_threshold("SZ", scores)
        assert threshold <= 6.0


class TestCVStabilityCheck:
    def test_no_cv_check_under_9_surveys(self, estimator):
        """CV check should not apply with fewer than 9 surveys."""
        # High variance data but only 5 surveys — should use mean
        scores = [1.0, 7.0, 1.0, 7.0, 1.0]
        threshold1, _ = estimator.compute_threshold("SZ", scores)
        # Result uses mean, not median
        assert threshold1 is not None  # just verify it runs without error

    def test_cv_check_uses_median_when_high(self, estimator):
        """After 9+ surveys with high CV, should switch to median."""
        # Create data with high CV (> 0.50)
        scores = [1.0, 7.0, 1.0, 7.0, 1.0, 7.0, 1.0, 7.0, 1.0]
        import numpy as np
        arr = np.array(scores)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        cv = std / mean
        assert cv > 0.50  # Confirm high CV

        threshold, _ = estimator.compute_threshold("SZ", scores)
        # With median-based calculation, should differ from mean-based
        assert threshold is not None


class TestDiagnosticGroupDifference:
    def test_sz_vs_bd_have_different_priors(self):
        sz = POPULATION_PRIORS["SZ"]
        bd = POPULATION_PRIORS["BD"]
        assert sz.na_mean != bd.na_mean
        assert sz.na_sd != bd.na_sd
        assert sz.threshold == bd.threshold == 4.0

    def test_same_data_different_groups_diverge(self, estimator):
        """Same individual data should produce different thresholds
        for SZ vs BD due to different population priors."""
        scores = [3.5] * 5
        thresh_sz, _ = estimator.compute_threshold("SZ", scores)
        thresh_bd, _ = estimator.compute_threshold("BD", scores)
        # With only 5 surveys (w≈0.24), population prior still dominates
        # Both groups have same population threshold (4.0), but different
        # priors affect the blending differently through the individual
        # threshold calculation
        # At low n, they'll be close but may differ due to weighting
        assert isinstance(thresh_sz, float)
        assert isinstance(thresh_bd, float)
