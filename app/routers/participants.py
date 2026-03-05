"""Participant management endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.participant import Participant, ParticipantPreference
from app.models.bandit import BanditArm
from app.models.threshold import ThresholdState
from app.schemas.participant import ParticipantCreate, ParticipantUpdate, ParticipantResponse
from app.services.thompson_sampling import create_fresh_arms

router = APIRouter(prefix="/api/v1/participants", tags=["participants"])


@router.post("", response_model=ParticipantResponse, status_code=201)
async def enroll_participant(
    data: ParticipantCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Enroll a new participant in the study.

    Creates the participant record, initializes all bandit arms with
    Beta(1,1) priors, and creates the threshold state with population prior.
    """
    # Create participant
    participant = Participant(
        study_id=data.study_id,
        diagnostic_group=data.diagnostic_group,
        condition=data.condition,
        wake_time=data.wake_time,
        sleep_time=data.sleep_time,
        timezone=data.timezone,
    )
    db.add(participant)
    await db.flush()

    # Create preferences
    prefs = ParticipantPreference(
        participant_id=participant.id,
        preferred_activities=data.preferred_activities,
    )
    db.add(prefs)

    # Initialize bandit arms for all three bandit systems
    for bandit_type in ["timing", "intervention_type", "boost"]:
        arms = create_fresh_arms(bandit_type)
        for arm in arms:
            db_arm = BanditArm(
                participant_id=participant.id,
                bandit_type=bandit_type,
                arm_name=arm.arm_name,
                arm_index=arm.arm_index,
                alpha=1.0,
                beta_param=1.0,
            )
            db.add(db_arm)

    # Initialize threshold state with population prior
    threshold = ThresholdState(
        participant_id=participant.id,
        current_threshold=4.0,  # population prior for both SZ and BD
    )
    db.add(threshold)

    await db.commit()
    await db.refresh(participant)
    return participant


@router.get("", response_model=list[ParticipantResponse])
async def list_participants(
    active_only: bool = True,
    db: AsyncSession = Depends(get_db),
):
    """List all participants (admin endpoint)."""
    query = select(Participant)
    if active_only:
        query = query.where(Participant.active == True)
    query = query.order_by(Participant.enrolled_at.desc())
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/{participant_id}", response_model=ParticipantResponse)
async def get_participant(
    participant_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a single participant by ID."""
    result = await db.execute(
        select(Participant).where(Participant.id == participant_id)
    )
    participant = result.scalar_one_or_none()
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
    return participant


@router.patch("/{participant_id}/settings", response_model=ParticipantResponse)
async def update_settings(
    participant_id: uuid.UUID,
    data: ParticipantUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update participant settings (wake/sleep time, timezone, active status)."""
    result = await db.execute(
        select(Participant).where(Participant.id == participant_id)
    )
    participant = result.scalar_one_or_none()
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")

    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(participant, key, value)
    participant.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(participant)
    return participant


@router.post("/{participant_id}/silence")
async def silence_surveys(
    participant_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Activate 30-minute silencing for a participant."""
    result = await db.execute(
        select(Participant).where(Participant.id == participant_id)
    )
    participant = result.scalar_one_or_none()
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")

    participant.silenced_until = datetime.utcnow() + timedelta(minutes=30)
    participant.updated_at = datetime.utcnow()
    await db.commit()

    return {"silenced_until": participant.silenced_until}
