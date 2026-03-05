"""Tests for the Thompson Sampling Engine."""

import pytest

from app.services.thompson_sampling import (
    ArmState,
    ThompsonSamplingEngine,
    create_fresh_arms,
    kl_beta,
)


def make_engine(n_arms: int = 3) -> ThompsonSamplingEngine:
    """Create an engine with n uniform-prior arms."""
    arms = [ArmState(f"arm_{i}", i) for i in range(n_arms)]
    return ThompsonSamplingEngine(arms)


class TestArmState:
    def test_initial_state(self):
        arm = ArmState("test", 0)
        assert arm.alpha == 1.0
        assert arm.beta == 1.0
        assert arm.times_selected == 0
        assert arm.total_reward == 0.0
        assert arm.mean_reward == 0.0
        assert arm.expected_value == 0.5  # Beta(1,1) mean = 0.5

    def test_mean_reward_after_updates(self):
        arm = ArmState("test", 0, times_selected=4, total_reward=3.0)
        assert arm.mean_reward == 0.75


class TestSelectArm:
    def test_returns_valid_index(self):
        engine = make_engine(4)
        for _ in range(100):
            idx, sample = engine.select_arm()
            assert 0 <= idx < 4
            assert 0.0 <= sample <= 1.0

    def test_excluded_arms_never_selected(self):
        engine = make_engine(4)
        excluded = [0, 1, 2]
        for _ in range(100):
            idx, _ = engine.select_arm(excluded_indices=excluded)
            assert idx == 3  # only arm 3 is available

    def test_biased_arm_selected_more(self):
        """An arm with strong prior should be selected more often."""
        arms = [
            ArmState("weak", 0, alpha=1.0, beta=10.0),
            ArmState("strong", 1, alpha=10.0, beta=1.0),
        ]
        engine = ThompsonSamplingEngine(arms)
        counts = {0: 0, 1: 0}
        for _ in range(1000):
            idx, _ = engine.select_arm()
            counts[idx] += 1
        assert counts[1] > counts[0]  # strong arm selected more


class TestUpdate:
    def test_reward_1_increments_alpha(self):
        engine = make_engine(2)
        engine.update(0, 1.0)
        arm = engine.arms[0]
        assert arm.alpha == 2.0
        assert arm.beta == 1.0
        assert arm.times_selected == 1
        assert arm.total_reward == 1.0

    def test_reward_0_increments_beta(self):
        engine = make_engine(2)
        engine.update(0, 0.0)
        arm = engine.arms[0]
        assert arm.alpha == 1.0
        assert arm.beta == 2.0
        assert arm.times_selected == 1
        assert arm.total_reward == 0.0

    def test_fractional_reward(self):
        engine = make_engine(2)
        engine.update(0, 0.75)
        arm = engine.arms[0]
        assert arm.alpha == 1.75
        assert arm.beta == 1.25

    def test_only_selected_arm_updated(self):
        engine = make_engine(3)
        engine.update(1, 1.0)
        assert engine.arms[0].alpha == 1.0
        assert engine.arms[0].beta == 1.0
        assert engine.arms[1].alpha == 2.0
        assert engine.arms[2].alpha == 1.0


class TestKLDivergence:
    def test_kl_zero_for_uniform(self):
        """KL(Beta(1,1) || Beta(1,1)) should be 0."""
        kl = kl_beta(1.0, 1.0, 1.0, 1.0)
        assert abs(kl) < 1e-10

    def test_kl_positive_after_updates(self):
        """KL from uniform should be positive after any update."""
        kl = kl_beta(2.0, 1.0, 1.0, 1.0)
        assert kl > 0

    def test_kl_increases_with_more_updates(self):
        """More updates should yield larger KL divergence."""
        kl_small = kl_beta(2.0, 2.0, 1.0, 1.0)
        kl_large = kl_beta(10.0, 2.0, 1.0, 1.0)
        assert kl_large > kl_small

    def test_engine_kl_per_arm(self):
        engine = make_engine(3)
        engine.update(0, 1.0)
        engine.update(0, 1.0)
        kl_values = engine.kl_divergence_from_uniform()
        assert kl_values["arm_0"] > 0
        assert abs(kl_values["arm_1"]) < 1e-10
        assert abs(kl_values["arm_2"]) < 1e-10


class TestConvergence:
    def test_not_converged_initially(self):
        engine = make_engine(3)
        assert not engine.has_converged()

    def test_converged_after_many_updates(self):
        engine = make_engine(3)
        for _ in range(20):
            engine.update(0, 0.8)
        assert engine.has_converged(threshold=0.10)

    def test_convergence_threshold_respected(self):
        engine = make_engine(2)
        engine.update(0, 1.0)
        # After one update, KL should be small
        kl = engine.kl_divergence_from_uniform()
        # Use a very high threshold that won't be met
        assert not engine.has_converged(threshold=100.0)


class TestCreateFreshArms:
    def test_timing_arms(self):
        arms = create_fresh_arms("timing")
        assert len(arms) == 4
        assert arms[0].arm_name == "immediate"
        assert arms[3].arm_name == "next_survey_enhancement"
        assert all(a.alpha == 1.0 and a.beta == 1.0 for a in arms)

    def test_intervention_type_arms(self):
        arms = create_fresh_arms("intervention_type")
        assert len(arms) == 4
        assert arms[3].arm_name == "social_outreach"

    def test_boost_arms(self):
        arms = create_fresh_arms("boost")
        assert len(arms) == 3
        assert "social_outreach" not in [a.arm_name for a in arms]

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            create_fresh_arms("unknown")
