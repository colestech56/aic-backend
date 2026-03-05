"""
Survey delivery and submission endpoints.

The survey submission endpoint is the CRITICAL entry point that triggers
the entire intervention pipeline.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.participant import Participant, ParticipantPreference
from app.models.survey import SurveySchedule, SurveyResponse
from app.models.threshold import ThresholdState
from app.schemas.survey import SurveySubmission, SurveyScheduleResponse, SurveyResponseOut
from app.services.bayesian_shrinkage import BayesianShrinkageEstimator
from app.services.intervention_engine import InterventionEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/surveys", tags=["surveys"])


@router.get("/pending", response_model=list[SurveyScheduleResponse])
async def get_pending_surveys(
    participant_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get all pending surveys for a participant."""
    now = datetime.utcnow()
    result = await db.execute(
        select(SurveySchedule)
        .where(and_(
            SurveySchedule.participant_id == participant_id,
            SurveySchedule.status.in_(["scheduled", "delivered"]),
            SurveySchedule.window_closes_at > now,
        ))
        .order_by(SurveySchedule.scheduled_at)
    )
    return result.scalars().all()


@router.post("/{schedule_id}/respond", response_model=SurveyResponseOut)
async def submit_survey(
    schedule_id: uuid.UUID,
    submission: SurveySubmission,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit a survey response. This is the critical endpoint that triggers
    the entire intervention pipeline for EMA surveys.

    Flow:
    1. Validate schedule
    2. Save response
    3. Compute NA score
    4. Update threshold state
    5. For EMA: evaluate intervention trigger
    6. For micro-survey: compute reward and update bandits
    """
    # Get the schedule
    result = await db.execute(
        select(SurveySchedule).where(SurveySchedule.id == schedule_id)
    )
    schedule = result.scalar_one_or_none()
    if not schedule:
        raise HTTPException(status_code=404, detail="Survey schedule not found")

    if schedule.status == "completed":
        raise HTTPException(status_code=400, detail="Survey already completed")

    if schedule.status == "expired":
        raise HTTPException(status_code=400, detail="Survey window has expired")

    # Check if window is still open
    now = datetime.utcnow()
    if now > schedule.window_closes_at:
        schedule.status = "expired"
        await db.commit()
        raise HTTPException(status_code=400, detail="Survey window has expired")

    # Compute NA score if anxiety and sadness are present
    response_data = submission.response_data
    na_score = None
    if "anxiety" in response_data and "sadness" in response_data:
        na_score = (response_data["anxiety"] + response_data["sadness"]) / 2.0

    # Save response
    survey_response = SurveyResponse(
        schedule_id=schedule_id,
        participant_id=schedule.participant_id,
        survey_type=schedule.survey_type,
        response_data=response_data,
        na_score=na_score,
    )
    db.add(survey_response)

    # Mark schedule as completed
    schedule.status = "completed"
    schedule.completed_at = now

    # Get participant info
    p_result = await db.execute(
        select(Participant).where(Participant.id == schedule.participant_id)
    )
    participant = p_result.scalar_one_or_none()
    if not participant:
        await db.rollback()
        raise HTTPException(status_code=404, detail="Participant not found")

    intervention_triggered = False
    intervention_id = None

    if schedule.survey_type == "ema" and na_score is not None:
        # Update threshold state — non-fatal if it fails
        try:
            await _update_threshold(db, schedule.participant_id, na_score, participant.diagnostic_group)
        except Exception:
            logger.exception("Failed to update threshold for participant %s", schedule.participant_id)

        # Evaluate intervention trigger (only for EMI condition)
        if participant.condition == "emi":
            try:
                # Get wellness preferences
                pref_result = await db.execute(
                    select(ParticipantPreference)
                    .where(ParticipantPreference.participant_id == participant.id)
                )
                prefs = pref_result.scalar_one_or_none()
                wellness_prefs = prefs.preferred_activities if prefs else []

                engine = InterventionEngine(db)
                intervention = await engine.evaluate_and_deliver(
                    participant_id=participant.id,
                    survey_response_id=survey_response.id,
                    na_score=na_score,
                    context={
                        "na_score": na_score,
                        "anxiety": response_data.get("anxiety"),
                        "sadness": response_data.get("sadness"),
                        "location": response_data.get("location", "unknown"),
                        "social_context": response_data.get("social_context", "unknown"),
                        "current_activity": response_data.get("current_activity", ""),
                    },
                    diagnostic_group=participant.diagnostic_group,
                    wellness_preferences=wellness_prefs,
                )
                if intervention:
                    intervention_triggered = True
                    intervention_id = intervention.id
            except Exception:
                logger.exception(
                    "Intervention pipeline failed for participant %s (survey response saved)",
                    schedule.participant_id,
                )

    elif schedule.survey_type == "micro" and schedule.linked_intervention_id:
        # Process micro-survey: compute reward and update bandits
        if na_score is not None and "helpfulness" in response_data:
            try:
                engine = InterventionEngine(db)
                await engine.process_micro_survey(
                    participant_id=schedule.participant_id,
                    intervention_id=schedule.linked_intervention_id,
                    na_post=na_score,
                    helpfulness=response_data["helpfulness"],
                )
            except Exception:
                logger.exception(
                    "Micro-survey reward processing failed for intervention %s",
                    schedule.linked_intervention_id,
                )

    elif schedule.survey_type == "end_of_day":
        # End-of-day: just store, no trigger evaluation
        pass

    await db.commit()

    return SurveyResponseOut(
        id=survey_response.id,
        survey_type=schedule.survey_type,
        na_score=na_score,
        submitted_at=survey_response.submitted_at,
        intervention_triggered=intervention_triggered,
        intervention_id=intervention_id,
    )


async def _update_threshold(
    db: AsyncSession,
    participant_id: uuid.UUID,
    na_score: float,
    diagnostic_group: str,
) -> None:
    """Update the participant's threshold state with a new NA score."""
    result = await db.execute(
        select(ThresholdState).where(ThresholdState.participant_id == participant_id)
    )
    state = result.scalar_one_or_none()

    if state is None:
        state = ThresholdState(participant_id=participant_id)
        db.add(state)

    # Append score and recalculate
    scores = list(state.na_scores) + [na_score]
    state.na_scores = scores
    state.n_surveys = len(scores)

    estimator = BayesianShrinkageEstimator()
    threshold, weight = estimator.compute_threshold(diagnostic_group, scores)

    state.current_threshold = threshold
    state.individual_weight = weight
    state.last_recalculated_at = datetime.utcnow()
    state.updated_at = datetime.utcnow()
