"""
Intervention Engine — the main orchestrator.

Ties together trigger evaluation, Thompson sampling, LLM content generation,
content screening, and delivery scheduling. Section 3.1 of the AIC spec.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.intervention import Intervention, BoostIntervention, DailyCounter
from app.models.bandit import BanditArm
from app.models.threshold import ThresholdState
from app.models.content import ContentHistory
from app.models.audit import AuditLog
from app.models.survey import SurveySchedule
from app.services.thompson_sampling import ThompsonSamplingEngine, ArmState
from app.services.bayesian_shrinkage import BayesianShrinkageEstimator
from app.services.reward_calculator import RewardCalculator
from app.services.llm_generator import LLMGenerator
from app.services.content_screener import ContentScreener
from app.services.static_fallbacks import get_fallback
from app.services.survey_scheduler import SurveyScheduler


# Timing arm → delivery delay in minutes
TIMING_DELAYS = {
    "immediate": 0,
    "5_min_delay": 5,
    "10_min_delay": 10,
    "next_survey_enhancement": None,  # handled separately
}


class InterventionEngine:
    """
    Main orchestration for intervention delivery.

    Called after each EMA survey submission to evaluate trigger conditions
    and manage the full intervention pipeline.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.shrinkage = BayesianShrinkageEstimator()
        self.reward_calc = RewardCalculator()
        self.llm = LLMGenerator()
        self.screener = ContentScreener()
        self.scheduler = SurveyScheduler()

    async def evaluate_and_deliver(
        self,
        participant_id: uuid.UUID,
        survey_response_id: uuid.UUID,
        na_score: float,
        context: dict,
        diagnostic_group: str,
        wellness_preferences: list[str] | None = None,
    ) -> Intervention | None:
        """
        Evaluate trigger conditions and deliver intervention if warranted.

        Steps (Section 3.1):
        1. Check NA ≥ threshold
        2. Check 2-hour cooldown
        3. Thompson-sample timing arm
        4. Thompson-sample intervention type arm (cap social_outreach at 2/day)
        5. Generate content via LLM
        6. Screen content
        7. Retry up to 2x or fallback to static
        8. Create intervention record
        9. Schedule micro-survey
        10. Log to audit trail
        """
        # Get current threshold
        threshold_state = await self._get_threshold_state(participant_id)
        threshold = threshold_state.current_threshold if threshold_state else 4.0

        # Step 1: Check trigger
        if na_score < threshold:
            await self._log_audit(participant_id, "trigger_not_met", {
                "na_score": na_score, "threshold": threshold,
            })
            return None

        # Step 2: Check cooldown
        if await self._in_cooldown(participant_id, "intervention"):
            await self._log_audit(participant_id, "cooldown_active", {
                "na_score": na_score, "threshold": threshold,
            })
            return None

        # Load bandit arms from DB
        timing_engine = await self._load_bandit(participant_id, "timing")
        type_engine = await self._load_bandit(participant_id, "intervention_type")

        # Step 3: Sample timing arm
        timing_idx, timing_sample = timing_engine.select_arm()
        timing_arm = timing_engine.get_arm_by_index(timing_idx)

        # Step 4: Sample intervention type arm (exclude social_outreach if at cap)
        excluded = await self._get_excluded_arms(participant_id)
        type_idx, type_sample = type_engine.select_arm(excluded_indices=excluded)
        type_arm = type_engine.get_arm_by_index(type_idx)

        # Steps 5-6: Generate and screen content
        content_text, content_source, screening_retries, cosine_sim = (
            await self._generate_screened_content(
                participant_id, type_arm.arm_name, context, wellness_preferences
            )
        )

        # Step 7: Calculate delivery time
        now = datetime.utcnow()
        delay = TIMING_DELAYS.get(timing_arm.arm_name)
        if delay is not None:
            delivery_at = now + timedelta(minutes=delay)
        else:
            delivery_at = None  # next_survey_enhancement — handled by survey system

        # Step 8: Create intervention record
        intervention = Intervention(
            participant_id=participant_id,
            trigger_survey_id=survey_response_id,
            trigger_na_score=na_score,
            threshold_at_trigger=threshold,
            timing_arm=timing_arm.arm_name,
            intervention_type_arm=type_arm.arm_name,
            timing_sample=timing_sample,
            type_sample=type_sample,
            content_text=content_text,
            content_source=content_source,
            screening_passed=content_source == "llm",
            screening_retries=screening_retries,
            cosine_sim_max=cosine_sim,
            scheduled_delivery_at=delivery_at,
            delivered_at=delivery_at if delay == 0 else None,
            status="delivered" if delay == 0 else "pending",
        )
        self.db.add(intervention)

        # Update daily counter
        await self._increment_daily_counter(participant_id, type_arm.arm_name)

        # Store content in history for dedup
        content_record = ContentHistory(
            participant_id=participant_id,
            intervention_type=type_arm.arm_name,
            content_text=content_text,
        )
        self.db.add(content_record)

        # Step 9: Schedule micro-survey (if immediate or delayed delivery)
        if delay is not None:
            actual_delivery = delivery_at
            micro = self.scheduler.schedule_micro_survey(actual_delivery)
            micro_schedule = SurveySchedule(
                participant_id=participant_id,
                survey_type="micro",
                scheduled_at=micro["scheduled_at"],
                window_closes_at=micro["window_closes_at"],
                linked_intervention_id=intervention.id,
            )
            self.db.add(micro_schedule)

        # Step 10: Audit log
        await self._log_audit(participant_id, "intervention_triggered", {
            "na_score": na_score,
            "threshold": threshold,
            "timing_arm": timing_arm.arm_name,
            "type_arm": type_arm.arm_name,
            "timing_sample": timing_sample,
            "type_sample": type_sample,
            "content_source": content_source,
            "screening_retries": screening_retries,
        })

        await self.db.flush()
        return intervention

    async def process_micro_survey(
        self,
        participant_id: uuid.UUID,
        intervention_id: uuid.UUID,
        na_post: float,
        helpfulness: float,
        is_boost: bool = False,
    ) -> float:
        """
        Process micro-survey response: compute reward and update bandits.

        Returns the combined reward.
        """
        if is_boost:
            return await self._process_boost_reward(
                participant_id, intervention_id, na_post, helpfulness
            )

        # Get the intervention record
        result = await self.db.execute(
            select(Intervention).where(Intervention.id == intervention_id)
        )
        intervention = result.scalar_one()

        # Compute reward
        reward = self.reward_calc.compute(
            na_pre=intervention.trigger_na_score,
            na_post=na_post,
            helpfulness=helpfulness,
        )

        # Update intervention record
        intervention.reward_na_change = reward.na_change_score
        intervention.reward_helpfulness = reward.helpfulness_score
        intervention.reward_combined = reward.combined_reward
        intervention.reward_calculated_at = datetime.utcnow()

        # Update bandit posteriors
        await self._update_bandit_arm(
            participant_id, "timing", intervention.timing_arm, reward.combined_reward
        )
        await self._update_bandit_arm(
            participant_id, "intervention_type",
            intervention.intervention_type_arm, reward.combined_reward
        )

        await self._log_audit(participant_id, "reward_computed", {
            "intervention_id": str(intervention_id),
            "na_pre": intervention.trigger_na_score,
            "na_post": na_post,
            "helpfulness": helpfulness,
            "reward_combined": reward.combined_reward,
        })

        await self.db.flush()
        return reward.combined_reward

    async def handle_boost(
        self,
        participant_id: uuid.UUID,
        na_score: float,
        context: dict,
        wellness_preferences: list[str] | None = None,
    ) -> BoostIntervention | None:
        """Handle a boost button press."""
        # Check boost cooldown (1 hour)
        if await self._in_cooldown(participant_id, "boost"):
            return None

        # Load boost bandit
        boost_engine = await self._load_bandit(participant_id, "boost")
        idx, sample = boost_engine.select_arm()
        arm = boost_engine.get_arm_by_index(idx)

        # Generate content
        content_text, content_source, _, _ = await self._generate_screened_content(
            participant_id, arm.arm_name, context, wellness_preferences
        )

        now = datetime.utcnow()
        boost = BoostIntervention(
            participant_id=participant_id,
            boost_arm=arm.arm_name,
            boost_sample=sample,
            content_text=content_text,
            content_source=content_source,
            trigger_na_score=na_score,
            delivered_at=now,
        )
        self.db.add(boost)

        # Schedule micro-survey
        micro = self.scheduler.schedule_micro_survey(now)
        micro_schedule = SurveySchedule(
            participant_id=participant_id,
            survey_type="micro",
            scheduled_at=micro["scheduled_at"],
            window_closes_at=micro["window_closes_at"],
            linked_intervention_id=boost.id,
        )
        self.db.add(micro_schedule)

        # Content history
        self.db.add(ContentHistory(
            participant_id=participant_id,
            intervention_type=arm.arm_name,
            content_text=content_text,
        ))

        await self._log_audit(participant_id, "boost_delivered", {
            "boost_arm": arm.arm_name,
            "na_score": na_score,
            "content_source": content_source,
        })

        await self.db.flush()
        return boost

    # --- Private helpers ---

    async def _get_threshold_state(self, participant_id: uuid.UUID) -> ThresholdState | None:
        result = await self.db.execute(
            select(ThresholdState).where(ThresholdState.participant_id == participant_id)
        )
        return result.scalar_one_or_none()

    async def _in_cooldown(self, participant_id: uuid.UUID, cooldown_type: str) -> bool:
        if cooldown_type == "intervention":
            cooldown_minutes = settings.INTERVENTION_COOLDOWN_MINUTES
            result = await self.db.execute(
                select(Intervention.delivered_at)
                .where(and_(
                    Intervention.participant_id == participant_id,
                    Intervention.delivered_at.is_not(None),
                ))
                .order_by(Intervention.delivered_at.desc())
                .limit(1)
            )
        else:  # boost
            cooldown_minutes = settings.BOOST_COOLDOWN_MINUTES
            result = await self.db.execute(
                select(BoostIntervention.delivered_at)
                .where(BoostIntervention.participant_id == participant_id)
                .order_by(BoostIntervention.delivered_at.desc())
                .limit(1)
            )

        last_delivery = result.scalar_one_or_none()
        if last_delivery is None:
            return False

        return datetime.utcnow() - last_delivery < timedelta(minutes=cooldown_minutes)

    async def _load_bandit(
        self, participant_id: uuid.UUID, bandit_type: str
    ) -> ThompsonSamplingEngine:
        result = await self.db.execute(
            select(BanditArm)
            .where(and_(
                BanditArm.participant_id == participant_id,
                BanditArm.bandit_type == bandit_type,
            ))
            .order_by(BanditArm.arm_index)
        )
        db_arms = result.scalars().all()

        arms = [
            ArmState(
                arm_name=a.arm_name,
                arm_index=a.arm_index,
                alpha=a.alpha,
                beta=a.beta_param,
                times_selected=a.times_selected,
                total_reward=a.total_reward,
            )
            for a in db_arms
        ]
        return ThompsonSamplingEngine(arms)

    async def _get_excluded_arms(self, participant_id: uuid.UUID) -> list[int]:
        """Get arm indices to exclude (social_outreach at daily cap)."""
        today = datetime.utcnow().date().isoformat()
        result = await self.db.execute(
            select(DailyCounter)
            .where(and_(
                DailyCounter.participant_id == participant_id,
                DailyCounter.counter_date == today,
            ))
        )
        counter = result.scalar_one_or_none()

        excluded = []
        if counter and counter.social_outreach_count >= settings.SOCIAL_OUTREACH_DAILY_CAP:
            excluded.append(3)  # social_outreach arm index
        return excluded

    async def _increment_daily_counter(
        self, participant_id: uuid.UUID, type_arm: str
    ) -> None:
        today = datetime.utcnow().date().isoformat()
        result = await self.db.execute(
            select(DailyCounter)
            .where(and_(
                DailyCounter.participant_id == participant_id,
                DailyCounter.counter_date == today,
            ))
        )
        counter = result.scalar_one_or_none()

        if counter is None:
            counter = DailyCounter(
                participant_id=participant_id,
                counter_date=today,
                intervention_count=0,
                social_outreach_count=0,
            )
            self.db.add(counter)

        counter.intervention_count = (counter.intervention_count or 0) + 1
        if type_arm == "social_outreach":
            counter.social_outreach_count = (counter.social_outreach_count or 0) + 1

    async def _generate_screened_content(
        self,
        participant_id: uuid.UUID,
        intervention_type: str,
        context: dict,
        wellness_preferences: list[str] | None,
    ) -> tuple[str, str, int, float | None]:
        """
        Generate and screen LLM content, with retry and static fallback.

        Returns (content_text, content_source, retries, max_cosine_sim).
        """
        # Get recent messages for dedup
        recent = await self._get_recent_content(participant_id)
        prior_today = await self._get_today_content(participant_id)

        max_retries = settings.MAX_LLM_RETRIES
        retries = 0
        max_sim = None

        for attempt in range(max_retries + 1):
            try:
                text = await self.llm.generate(
                    intervention_type=intervention_type,
                    context=context,
                    prior_messages_today=prior_today,
                    wellness_preferences=wellness_preferences,
                )

                passed, violations = self.screener.screen(text, recent)
                # Track cosine similarity if present in violations
                for v in violations:
                    if v.startswith("too_similar:"):
                        max_sim = float(v.split(":")[1])

                if passed:
                    return (text, "llm", retries, max_sim)

                retries += 1
                await self._log_audit(participant_id, "content_screening_failed", {
                    "attempt": attempt + 1,
                    "violations": violations,
                    "text_preview": text[:100],
                })

            except Exception as e:
                retries += 1
                await self._log_audit(participant_id, "llm_generation_failed", {
                    "attempt": attempt + 1,
                    "error": str(e),
                })

        # All retries exhausted — use static fallback
        fallback = get_fallback(intervention_type)
        return (fallback, "static_fallback", retries, max_sim)

    async def _get_recent_content(self, participant_id: uuid.UUID) -> list[str]:
        """Get content delivered in the past 24 hours for dedup."""
        cutoff = datetime.utcnow() - timedelta(hours=24)
        result = await self.db.execute(
            select(ContentHistory.content_text)
            .where(and_(
                ContentHistory.participant_id == participant_id,
                ContentHistory.delivered_at >= cutoff,
            ))
            .order_by(ContentHistory.delivered_at.desc())
        )
        return [r[0] for r in result.all()]

    async def _get_today_content(self, participant_id: uuid.UUID) -> list[str]:
        """Get all content delivered today for LLM context."""
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        result = await self.db.execute(
            select(ContentHistory.content_text)
            .where(and_(
                ContentHistory.participant_id == participant_id,
                ContentHistory.delivered_at >= today_start,
            ))
        )
        return [r[0] for r in result.all()]

    async def _update_bandit_arm(
        self,
        participant_id: uuid.UUID,
        bandit_type: str,
        arm_name: str,
        reward: float,
    ) -> None:
        """Update a single bandit arm's posterior with row-level locking."""
        result = await self.db.execute(
            select(BanditArm)
            .where(and_(
                BanditArm.participant_id == participant_id,
                BanditArm.bandit_type == bandit_type,
                BanditArm.arm_name == arm_name,
            ))
            .with_for_update()
        )
        arm = result.scalar_one()
        arm.alpha += reward
        arm.beta_param += (1.0 - reward)
        arm.times_selected += 1
        arm.total_reward += reward
        arm.updated_at = datetime.utcnow()

    async def _process_boost_reward(
        self,
        participant_id: uuid.UUID,
        boost_id: uuid.UUID,
        na_post: float,
        helpfulness: float,
    ) -> float:
        result = await self.db.execute(
            select(BoostIntervention).where(BoostIntervention.id == boost_id)
        )
        boost = result.scalar_one()

        reward = self.reward_calc.compute(
            na_pre=boost.trigger_na_score,
            na_post=na_post,
            helpfulness=helpfulness,
        )

        boost.reward_na_change = reward.na_change_score
        boost.reward_helpfulness = reward.helpfulness_score
        boost.reward_combined = reward.combined_reward
        boost.reward_calculated_at = datetime.utcnow()

        await self._update_bandit_arm(
            participant_id, "boost", boost.boost_arm, reward.combined_reward
        )

        await self.db.flush()
        return reward.combined_reward

    async def _log_audit(
        self, participant_id: uuid.UUID, event_type: str, event_data: dict
    ) -> None:
        self.db.add(AuditLog(
            participant_id=participant_id,
            event_type=event_type,
            event_data=event_data,
        ))
