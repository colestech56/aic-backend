"""Tests for the Reward Calculator."""

import pytest

from app.services.reward_calculator import RewardCalculator


@pytest.fixture
def calc():
    return RewardCalculator()


class TestNAChangeReward:
    def test_large_decrease_reward_1_0(self, calc):
        """NA reduced by 50%+ of possible range → reward 1.0."""
        # NA_pre=5, NA_post=2: change=(5-2)/(5-1)=0.75 ≥ 0.5
        result = calc.compute(na_pre=5.0, na_post=2.0, helpfulness=None)
        assert result.na_change_score == 1.0

    def test_moderate_decrease_reward_0_75(self, calc):
        """NA reduced by 25-49% → reward 0.75."""
        # NA_pre=5, NA_post=3.5: change=(5-3.5)/(5-1)=0.375 ≥ 0.25
        result = calc.compute(na_pre=5.0, na_post=3.5, helpfulness=None)
        assert result.na_change_score == 0.75

    def test_no_change_reward_0_5(self, calc):
        """No change or slight reduction → reward 0.5."""
        result = calc.compute(na_pre=5.0, na_post=5.0, helpfulness=None)
        assert result.na_change_score == 0.5

    def test_slight_decrease_reward_0_5(self, calc):
        """Small positive change (< 0.25 normalized) → reward 0.5."""
        # NA_pre=5, NA_post=4.5: change=(5-4.5)/(5-1)=0.125 ≥ 0
        result = calc.compute(na_pre=5.0, na_post=4.5, helpfulness=None)
        assert result.na_change_score == 0.5

    def test_na_increase_reward_0_2(self, calc):
        """NA increased → reward 0.2."""
        # NA_pre=4, NA_post=5: change=(4-5)/(4-1)=-0.33 < 0
        result = calc.compute(na_pre=4.0, na_post=5.0, helpfulness=None)
        assert result.na_change_score == 0.2

    def test_na_at_floor_neutral(self, calc):
        """NA_pre=1 (shouldn't have triggered) → neutral 0.5."""
        result = calc.compute(na_pre=1.0, na_post=1.0, helpfulness=None)
        assert result.na_change_score == 0.5

    def test_missing_micro_survey_neutral(self, calc):
        """Missing post-NA → neutral 0.5."""
        result = calc.compute(na_pre=5.0, na_post=None, helpfulness=None)
        assert result.na_change_score == 0.5


class TestHelpfulnessReward:
    def test_max_helpfulness(self, calc):
        """Helpfulness=5 → normalized to 1.0."""
        result = calc.compute(na_pre=5.0, na_post=5.0, helpfulness=5)
        assert result.helpfulness_score == 1.0

    def test_min_helpfulness(self, calc):
        """Helpfulness=1 → normalized to 0.0."""
        result = calc.compute(na_pre=5.0, na_post=5.0, helpfulness=1)
        assert result.helpfulness_score == 0.0

    def test_mid_helpfulness(self, calc):
        """Helpfulness=3 → normalized to 0.5."""
        result = calc.compute(na_pre=5.0, na_post=5.0, helpfulness=3)
        assert result.helpfulness_score == 0.5

    def test_missing_helpfulness_neutral(self, calc):
        """Missing helpfulness → 0.5."""
        result = calc.compute(na_pre=5.0, na_post=3.0, helpfulness=None)
        assert result.helpfulness_score == 0.5


class TestCombinedReward:
    def test_weights_70_30(self, calc):
        """Combined reward should be 70% NA + 30% helpfulness."""
        # NA_pre=5, NA_post=2: na_reward=1.0
        # Helpfulness=5: help_reward=1.0
        # Combined = 0.70*1.0 + 0.30*1.0 = 1.0
        result = calc.compute(na_pre=5.0, na_post=2.0, helpfulness=5)
        assert abs(result.combined_reward - 1.0) < 1e-10

    def test_mixed_rewards(self, calc):
        """Test with different NA and helpfulness outcomes."""
        # NA_pre=5, NA_post=5: na_reward=0.5 (no change)
        # Helpfulness=1: help_reward=0.0
        # Combined = 0.70*0.5 + 0.30*0.0 = 0.35
        result = calc.compute(na_pre=5.0, na_post=5.0, helpfulness=1)
        assert abs(result.combined_reward - 0.35) < 1e-10

    def test_all_missing_gives_neutral(self, calc):
        """Both missing → neutral for both → combined neutral."""
        result = calc.compute(na_pre=5.0, na_post=None, helpfulness=None)
        # 0.70*0.5 + 0.30*0.5 = 0.5
        assert abs(result.combined_reward - 0.5) < 1e-10

    def test_normalized_change_stored(self, calc):
        """Normalized change value should be stored in result."""
        result = calc.compute(na_pre=5.0, na_post=2.0, helpfulness=3)
        assert result.normalized_change is not None
        assert abs(result.normalized_change - 0.75) < 1e-10
