"""Admin dashboard endpoints for researchers."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, and_, case
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.participant import Participant
from app.models.bandit import BanditArm
from app.models.threshold import ThresholdState
from app.models.survey import SurveySchedule, SurveyResponse
from app.models.intervention import Intervention, BoostIntervention
from app.models.audit import AuditLog
from app.schemas.intervention import (
    BanditArmState,
    BanditStateResponse,
    ThresholdStateResponse,
    InterventionResponse,
)
from app.services.thompson_sampling import kl_beta

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.get("/dashboard")
async def dashboard_overview(
    db: AsyncSession = Depends(get_db),
):
    """Summary stats for the admin dashboard."""
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Active participants
    active_count = await db.execute(
        select(func.count()).where(Participant.active == True)
    )
    total_active = active_count.scalar()

    # Surveys completed today
    surveys_today = await db.execute(
        select(func.count()).select_from(SurveyResponse)
        .where(SurveyResponse.submitted_at >= today_start)
    )
    total_surveys_today = surveys_today.scalar()

    # Interventions triggered today
    interventions_today = await db.execute(
        select(func.count()).select_from(Intervention)
        .where(Intervention.created_at >= today_start)
    )
    total_interventions_today = interventions_today.scalar()

    # Average NA score today
    avg_na = await db.execute(
        select(func.avg(SurveyResponse.na_score))
        .where(and_(
            SurveyResponse.submitted_at >= today_start,
            SurveyResponse.na_score.is_not(None),
        ))
    )
    avg_na_today = avg_na.scalar()

    # Boosts today
    boosts_today = await db.execute(
        select(func.count()).select_from(BoostIntervention)
        .where(BoostIntervention.delivered_at >= today_start)
    )
    total_boosts_today = boosts_today.scalar()

    return {
        "active_participants": total_active,
        "surveys_completed_today": total_surveys_today,
        "interventions_triggered_today": total_interventions_today,
        "boosts_delivered_today": total_boosts_today,
        "average_na_today": round(avg_na_today, 2) if avg_na_today else None,
        "timestamp": now.isoformat(),
    }


@router.get("/participants/{participant_id}/bandits")
async def get_bandit_state(
    participant_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get full bandit state for a participant."""
    result = await db.execute(
        select(BanditArm)
        .where(BanditArm.participant_id == participant_id)
        .order_by(BanditArm.bandit_type, BanditArm.arm_index)
    )
    arms = result.scalars().all()

    if not arms:
        raise HTTPException(status_code=404, detail="No bandit arms found")

    # Group by bandit type
    bandits = {}
    for arm in arms:
        if arm.bandit_type not in bandits:
            bandits[arm.bandit_type] = []

        kl = kl_beta(arm.alpha, arm.beta_param, 1.0, 1.0)
        bandits[arm.bandit_type].append(BanditArmState(
            arm_name=arm.arm_name,
            arm_index=arm.arm_index,
            alpha=arm.alpha,
            beta=arm.beta_param,
            times_selected=arm.times_selected,
            total_reward=arm.total_reward,
            mean_reward=arm.total_reward / arm.times_selected if arm.times_selected > 0 else 0,
            kl_divergence=kl,
        ))

    responses = []
    for bandit_type, arm_states in bandits.items():
        converged = any(a.kl_divergence > 0.10 for a in arm_states)
        responses.append(BanditStateResponse(
            participant_id=participant_id,
            bandit_type=bandit_type,
            arms=arm_states,
            converged=converged,
        ))

    return responses


@router.get("/participants/{participant_id}/threshold")
async def get_threshold_state(
    participant_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get threshold state for a participant."""
    result = await db.execute(
        select(ThresholdState).where(ThresholdState.participant_id == participant_id)
    )
    state = result.scalar_one_or_none()
    if not state:
        raise HTTPException(status_code=404, detail="Threshold state not found")

    # Get diagnostic group
    p_result = await db.execute(
        select(Participant.diagnostic_group).where(Participant.id == participant_id)
    )
    diag = p_result.scalar_one_or_none()

    return ThresholdStateResponse(
        participant_id=participant_id,
        current_threshold=state.current_threshold,
        individual_weight=state.individual_weight,
        n_surveys=state.n_surveys,
        diagnostic_group=diag or "unknown",
        population_prior=4.0,
    )


@router.get("/participants/{participant_id}/interventions")
async def get_participant_interventions(
    participant_id: uuid.UUID,
    limit: int = Query(default=50, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Get recent interventions for a participant."""
    result = await db.execute(
        select(Intervention)
        .where(Intervention.participant_id == participant_id)
        .order_by(Intervention.created_at.desc())
        .limit(limit)
    )
    interventions = result.scalars().all()
    return [InterventionResponse.model_validate(i) for i in interventions]


@router.get("/convergence")
async def convergence_overview(
    db: AsyncSession = Depends(get_db),
):
    """
    Get convergence status for all active participants.

    Returns KL divergence from uniform prior for each arm of each bandit
    for each participant.
    """
    # Get all active participants
    p_result = await db.execute(
        select(Participant.id, Participant.study_id)
        .where(Participant.active == True)
    )
    participants = p_result.all()

    convergence_data = []
    for p_id, study_id in participants:
        arms_result = await db.execute(
            select(BanditArm)
            .where(BanditArm.participant_id == p_id)
            .order_by(BanditArm.bandit_type, BanditArm.arm_index)
        )
        arms = arms_result.scalars().all()

        participant_data = {
            "participant_id": str(p_id),
            "study_id": study_id,
            "bandits": {},
        }

        for arm in arms:
            if arm.bandit_type not in participant_data["bandits"]:
                participant_data["bandits"][arm.bandit_type] = {
                    "arms": [],
                    "converged": False,
                }
            kl = kl_beta(arm.alpha, arm.beta_param, 1.0, 1.0)
            participant_data["bandits"][arm.bandit_type]["arms"].append({
                "arm_name": arm.arm_name,
                "kl_divergence": round(kl, 4),
                "times_selected": arm.times_selected,
            })
            if kl > 0.10:
                participant_data["bandits"][arm.bandit_type]["converged"] = True

        convergence_data.append(participant_data)

    return convergence_data


@router.get("/audit")
async def get_audit_log(
    participant_id: uuid.UUID | None = None,
    event_type: str | None = None,
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """Filterable, paginated audit log."""
    query = select(AuditLog).order_by(AuditLog.created_at.desc())

    if participant_id:
        query = query.where(AuditLog.participant_id == participant_id)
    if event_type:
        query = query.where(AuditLog.event_type == event_type)

    query = query.offset(offset).limit(limit)
    result = await db.execute(query)
    logs = result.scalars().all()

    return [
        {
            "id": log.id,
            "participant_id": str(log.participant_id) if log.participant_id else None,
            "event_type": log.event_type,
            "event_data": log.event_data,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]
