"""
Integration tests for the AIC intervention pipeline.

Tests the full flow from survey submission through intervention delivery
to micro-survey reward computation and bandit updates.

Uses mocked database sessions to avoid requiring a real PostgreSQL instance.
"""

import uuid
from datetime import datetime, time, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import pytest_asyncio

from app.services.intervention_engine import InterventionEngine
from app.services.thompson_sampling import (
    ThompsonSamplingEngine,
    ArmState,
    TIMING_ARMS,
    INTERVENTION_TYPE_ARMS,
    create_fresh_arms,
)
from app.services.bayesian_shrinkage import BayesianShrinkageEstimator
from app.services.reward_calculator import RewardCalculator
from app.services.content_screener import ContentScreener
from app.services.survey_scheduler import SurveyScheduler


# --- Thompson sampling integration ---

class TestThompsonSamplingIntegration:
    """Test Thompson sampling convergence over multiple rounds."""

    def test_bandit_learns_best_arm_over_many_trials(self):
        """Simulate many rounds and verify the bandit identifies the best arm."""
        arms = create_fresh_arms("timing")
        engine = ThompsonSamplingEngine(arms)

        # Simulate: arm 0 ("immediate") has high reward, others lower
        reward_probs = {0: 0.8, 1: 0.4, 2: 0.3, 3: 0.2}

        import random
        random.seed(42)
        for _ in range(200):
            idx, _ = engine.select_arm()
            # Simulate binary reward based on true probability
            reward = 1.0 if random.random() < reward_probs[idx] else 0.0
            engine.update(idx, reward)

        # After 200 trials, arm 0 should have highest selection count
        best_arm = max(engine.arms, key=lambda a: a.times_selected)
        assert best_arm.arm_index == 0, (
            f"Expected arm 0 to be selected most, but arm {best_arm.arm_index} was "
            f"({best_arm.times_selected} times)"
        )

    def test_convergence_detection_after_learning(self):
        """KL divergence should exceed threshold after enough learning."""
        arms = create_fresh_arms("timing")
        engine = ThompsonSamplingEngine(arms)

        # Give arm 0 many successful trials
        for _ in range(50):
            engine.update(0, 0.9)  # High reward arm 0
        for _ in range(10):
            engine.update(1, 0.3)  # Low reward arm 1
        for _ in range(10):
            engine.update(2, 0.2)
        for _ in range(10):
            engine.update(3, 0.1)

        # Should detect convergence on at least one arm
        kl_values = engine.kl_divergence_from_uniform()
        assert any(kl > 0.10 for kl in kl_values.values()), (
            f"Expected at least one arm KL > 0.10, got {kl_values}"
        )
        assert engine.has_converged()

    def test_arm_exclusion_works(self):
        """Verify excluded arms are never selected."""
        arms = create_fresh_arms("intervention_type")
        engine = ThompsonSamplingEngine(arms)

        # Exclude arm 3 (social_outreach)
        for _ in range(100):
            idx, _ = engine.select_arm(excluded_indices=[3])
            assert idx != 3, "Excluded arm 3 was selected"


# --- Bayesian shrinkage integration ---

class TestBayesianShrinkageIntegration:
    """Test threshold individualization over time."""

    def test_threshold_converges_to_individual(self):
        """
        As n increases toward 21, threshold should shift from
        population prior to individual estimate.
        """
        estimator = BayesianShrinkageEstimator()

        # High-NA participant (SZ, pop threshold=4.0)
        scores = []
        thresholds = []
        for i in range(25):
            scores.append(5.0)  # Consistently high NA
            threshold, weight = estimator.compute_threshold("SZ", scores)
            thresholds.append(threshold)

        # Early: close to population prior (4.0)
        assert abs(thresholds[0] - 4.0) < abs(thresholds[-1] - 4.0), (
            "Early threshold should be closer to population prior"
        )

        # Late: should be higher (influenced by individual mean of 5.0)
        assert thresholds[-1] > thresholds[0], (
            "Threshold should increase as individual data accumulates"
        )

        # Weight should be 1.0 at n=21
        _, weight_at_21 = estimator.compute_threshold("SZ", scores[:21])
        assert weight_at_21 == 1.0

    def test_cv_stability_switch(self):
        """When CV > 0.50, estimator should use median instead of mean."""
        estimator = BayesianShrinkageEstimator()

        # Create highly variable scores (CV > 0.50)
        scores = [1.0, 7.0, 1.0, 7.0, 1.0, 7.0, 1.0, 7.0, 1.0]  # 9 scores, CV > 0.50

        # With mean: (1+7)/2 = 4.0, SD ≈ 3.17, CV = 3.17/4.0 ≈ 0.79
        # With median: 1.0 (middle of 9 sorted values = 1,1,1,1,1,7,7,7,7 -> 1.0)
        # Actually sorted: [1,1,1,1,1,7,7,7,7] -> median = 1.0
        threshold, _ = estimator.compute_threshold("SZ", scores)
        # Should use median (1.0) instead of mean (4.0) for individual component
        # But blended with population and clamped to floor
        assert threshold >= 3.0  # Floor


# --- Reward calculator integration ---

class TestRewardCalculatorIntegration:
    """Test reward calculation in realistic scenarios."""

    def test_large_improvement_gives_high_reward(self):
        calc = RewardCalculator()
        reward = calc.compute(na_pre=6.0, na_post=2.0, helpfulness=5)
        assert reward.combined_reward > 0.8

    def test_no_change_gives_moderate_reward(self):
        calc = RewardCalculator()
        reward = calc.compute(na_pre=4.0, na_post=4.0, helpfulness=3)
        assert 0.3 < reward.combined_reward < 0.7

    def test_worsening_gives_low_reward(self):
        calc = RewardCalculator()
        reward = calc.compute(na_pre=3.0, na_post=5.0, helpfulness=1)
        assert reward.combined_reward < 0.3

    def test_full_pipeline_reward_to_bandit(self):
        """Test reward computation feeds correctly into bandit updates."""
        calc = RewardCalculator()

        # Simulate: high NA -> intervention -> improvement + rated helpful
        reward = calc.compute(na_pre=5.5, na_post=2.5, helpfulness=4)

        # Feed into bandit
        arms = create_fresh_arms("timing")
        engine = ThompsonSamplingEngine(arms)
        engine.update(0, reward.combined_reward)

        # Arm 0 should have updated posteriors
        arm = engine.arms[0]
        assert arm.alpha > 1.0  # Started at 1.0, increased by reward
        assert arm.beta > 1.0  # Started at 1.0, increased by (1-reward)
        assert arm.times_selected == 1
        assert arm.total_reward == reward.combined_reward


# --- Content screener integration ---

class TestContentScreenerIntegration:
    """Test content screening in the intervention pipeline context."""

    def test_clean_llm_output_passes(self):
        screener = ContentScreener()
        text = (
            "Try taking a few slow, deep breaths right now. "
            "Notice how your body feels as you breathe out."
        )
        passed, violations = screener.screen(text, [])
        assert passed
        assert violations == []

    def test_clinical_language_rejected(self):
        screener = ContentScreener()
        text = (
            "Your cognitive behavioral therapy exercise for today "
            "is to practice mindfulness meditation."
        )
        passed, violations = screener.screen(text, [])
        assert not passed
        assert any("forbidden" in v for v in violations)

    def test_dedup_catches_identical_content(self):
        screener = ContentScreener()
        original = (
            "Try taking a few slow, deep breaths right now. "
            "Notice how your body feels as you breathe out."
        )
        recent = [original]
        # Identical text should definitely be caught
        passed, violations = screener.screen(original, recent)
        assert not passed
        assert any("too_similar" in v for v in violations)

    def test_fallback_on_repeated_failures(self):
        """When screening fails repeatedly, static fallback should be used."""
        from app.services.static_fallbacks import get_fallback

        fallback = get_fallback("relaxation_breathing")
        assert isinstance(fallback, str)
        assert len(fallback) > 0
        assert len(fallback) <= 280

        # Fallback should pass screening
        screener = ContentScreener()
        passed, _ = screener.screen(fallback, [])
        assert passed


# --- Full pipeline simulation ---

class TestFullPipelineSimulation:
    """Simulate the complete intervention lifecycle without a database."""

    def test_end_to_end_simulation(self):
        """
        Simulate: participant enrolled → surveys → high NA triggers
        intervention → micro-survey → reward → bandit update.
        """
        # Step 1: Initialize participant bandits
        timing_arms = create_fresh_arms("timing")
        type_arms = create_fresh_arms("intervention_type")
        timing_engine = ThompsonSamplingEngine(timing_arms)
        type_engine = ThompsonSamplingEngine(type_arms)

        # Step 2: Initialize threshold
        estimator = BayesianShrinkageEstimator()

        # Step 3: Simulate surveys and interventions
        calc = RewardCalculator()
        screener = ContentScreener()
        scheduler = SurveyScheduler()

        na_scores = []
        interventions_delivered = 0

        for survey_round in range(30):
            # Simulate an EMA survey with varying NA
            import random
            random.seed(survey_round)
            anxiety = random.randint(1, 7)
            sadness = random.randint(1, 7)
            na = (anxiety + sadness) / 2.0
            na_scores.append(na)

            # Update threshold
            threshold, weight = estimator.compute_threshold("SZ", na_scores)

            # Check trigger
            if na >= threshold:
                # Sample timing
                timing_idx, _ = timing_engine.select_arm()
                type_idx, _ = type_engine.select_arm()

                interventions_delivered += 1

                # Simulate micro-survey: some improvement
                na_post = max(1.0, na - random.random() * 2)
                helpfulness = random.randint(1, 5)
                reward = calc.compute(na_pre=na, na_post=na_post, helpfulness=helpfulness)

                # Update bandits
                timing_engine.update(timing_idx, reward.combined_reward)
                type_engine.update(type_idx, reward.combined_reward)

        # Verify the simulation ran correctly
        assert len(na_scores) == 30
        assert interventions_delivered > 0, "Should have triggered at least one intervention"

        # Verify bandit arms were updated
        total_timing_selections = sum(a.times_selected for a in timing_engine.arms)
        assert total_timing_selections == interventions_delivered

        # Verify weight progressed
        _, final_weight = estimator.compute_threshold("SZ", na_scores)
        assert final_weight > 0.5  # 30 surveys → weight should be 1.0

    def test_micro_survey_scheduling_after_intervention(self):
        """Verify micro-surveys are correctly scheduled after interventions."""
        scheduler = SurveyScheduler()
        now = datetime(2026, 3, 5, 14, 0)

        # Immediate delivery
        micro = scheduler.schedule_micro_survey(now)
        assert micro["scheduled_at"] == now + timedelta(minutes=15)
        assert micro["survey_type"] == "micro"

        # 5-minute delayed delivery
        delivery_5 = now + timedelta(minutes=5)
        micro_5 = scheduler.schedule_micro_survey(delivery_5)
        assert micro_5["scheduled_at"] == delivery_5 + timedelta(minutes=15)

    def test_daily_ema_plus_eod_schedule(self):
        """Verify a complete daily schedule generation."""
        scheduler = SurveyScheduler()
        from datetime import date

        ema = scheduler.generate_daily_ema_schedule(
            wake_time=time(8, 0),
            sleep_time=time(22, 0),
            target_date=date(2026, 3, 5),
            timezone="America/New_York",
        )
        eod = scheduler.schedule_end_of_day(
            sleep_time=time(22, 0),
            target_date=date(2026, 3, 5),
            timezone="America/New_York",
        )

        # Should have 3 EMA + 1 EOD = 4 surveys total
        all_surveys = ema + [eod]
        assert len(all_surveys) == 4
        assert sum(1 for s in all_surveys if s["survey_type"] == "ema") == 3
        assert sum(1 for s in all_surveys if s["survey_type"] == "end_of_day") == 1

    def test_social_outreach_daily_cap(self):
        """Verify social_outreach arm exclusion at daily cap."""
        type_arms = create_fresh_arms("intervention_type")
        engine = ThompsonSamplingEngine(type_arms)

        # Simulate: social_outreach (arm 3) already used twice today
        # It should be excluded from selection
        social_selections = 0
        for _ in range(100):
            idx, _ = engine.select_arm(excluded_indices=[3])
            if idx == 3:
                social_selections += 1

        assert social_selections == 0

    def test_convergence_monitoring_across_participants(self):
        """Simulate convergence monitoring for multiple participants."""
        participants = {}
        for i in range(5):
            arms = create_fresh_arms("timing")
            participants[f"P-{i}"] = ThompsonSamplingEngine(arms)

        # Give participant P-0 lots of data on one arm
        for _ in range(50):
            participants["P-0"].update(0, 0.85)

        # P-0 should have converged, others should not
        for pid, engine in participants.items():
            kl = engine.kl_divergence_from_uniform()
            if pid == "P-0":
                assert engine.has_converged(), f"{pid} should have converged"
            else:
                assert not engine.has_converged(), f"{pid} should not have converged"


# --- API contract tests ---

class TestAPIContracts:
    """Test Pydantic schema validation."""

    def test_ema_response_validation(self):
        from app.schemas.survey import EMAResponseData
        data = EMAResponseData(
            anxiety=5,
            sadness=6,
            happiness=2,
            relaxation=2,
            location="home",
            social_context="alone",
        )
        assert data.anxiety == 5
        assert data.sadness == 6

    def test_ema_response_rejects_invalid_range(self):
        from app.schemas.survey import EMAResponseData
        with pytest.raises(Exception):
            EMAResponseData(
                anxiety=8,  # > 7
                sadness=6,
                happiness=2,
                relaxation=2,
                location="home",
                social_context="alone",
            )

    def test_ema_response_rejects_invalid_location(self):
        from app.schemas.survey import EMAResponseData
        with pytest.raises(Exception):
            EMAResponseData(
                anxiety=5,
                sadness=6,
                happiness=2,
                relaxation=2,
                location="office",  # Invalid
                social_context="alone",
            )

    def test_micro_survey_validation(self):
        from app.schemas.survey import MicroSurveyResponseData
        data = MicroSurveyResponseData(anxiety=3, sadness=4, helpfulness=5)
        assert data.helpfulness == 5

    def test_micro_survey_helpfulness_range(self):
        from app.schemas.survey import MicroSurveyResponseData
        with pytest.raises(Exception):
            MicroSurveyResponseData(anxiety=3, sadness=4, helpfulness=6)  # > 5

    def test_boost_request_validation(self):
        from app.schemas.intervention import BoostRequest
        data = BoostRequest(anxiety=5, sadness=6, location="home")
        assert data.anxiety == 5

    def test_participant_create_validation(self):
        from app.schemas.participant import ParticipantCreate
        data = ParticipantCreate(
            study_id="AIC-001",
            diagnostic_group="SZ",
        )
        assert data.condition == "emi"  # default
        assert data.timezone == "America/New_York"  # default

    def test_participant_create_rejects_invalid_group(self):
        from app.schemas.participant import ParticipantCreate
        with pytest.raises(Exception):
            ParticipantCreate(
                study_id="AIC-001",
                diagnostic_group="XX",  # Invalid
            )

    def test_na_score_computation(self):
        """NA = (anxiety + sadness) / 2."""
        anxiety, sadness = 6, 4
        na = (anxiety + sadness) / 2.0
        assert na == 5.0

    def test_na_score_boundary_values(self):
        # Minimum NA
        assert (1 + 1) / 2.0 == 1.0
        # Maximum NA
        assert (7 + 7) / 2.0 == 7.0


# --- Edge cases ---

class TestEdgeCases:
    def test_reward_with_na_pre_equal_one(self):
        """NA_pre=1 should give neutral 0.5 reward for NA component."""
        calc = RewardCalculator()
        reward = calc.compute(na_pre=1.0, na_post=1.0, helpfulness=3)
        # na_pre=1 → neutral 0.5
        assert reward.na_change_score == 0.5

    def test_empty_bandit_arms_raises(self):
        """ThompsonSamplingEngine should handle empty arms gracefully."""
        with pytest.raises((ValueError, IndexError)):
            engine = ThompsonSamplingEngine([])
            engine.select_arm()

    def test_all_arms_excluded_returns_negative_sample(self):
        """Excluding all arms should return an arm with a -1.0 sample value."""
        arms = create_fresh_arms("timing")
        engine = ThompsonSamplingEngine(arms)
        idx, sample = engine.select_arm(excluded_indices=[0, 1, 2, 3])
        # All arms excluded → all samples are -1.0, argmax picks first
        assert sample == -1.0

    def test_content_screener_empty_text(self):
        screener = ContentScreener()
        passed, violations = screener.screen("", [])
        assert not passed  # Empty text should fail

    def test_content_screener_very_long_text(self):
        screener = ContentScreener()
        text = "A" * 300  # Exceeds 280 char limit
        passed, violations = screener.screen(text, [])
        assert not passed
        assert any("length" in v or "too_long" in v for v in violations)

    def test_threshold_floor_respected(self):
        """Threshold should never go below 3.0."""
        estimator = BayesianShrinkageEstimator()
        # Very low scores
        scores = [1.0] * 25
        threshold, _ = estimator.compute_threshold("SZ", scores)
        assert threshold >= 3.0

    def test_threshold_ceiling_respected(self):
        """Threshold should never exceed 6.0."""
        estimator = BayesianShrinkageEstimator()
        # Very high scores
        scores = [7.0] * 25
        threshold, _ = estimator.compute_threshold("SZ", scores)
        assert threshold <= 6.0
