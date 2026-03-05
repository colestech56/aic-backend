"""Boost (on-demand support) endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.participant import Participant, ParticipantPreference
from app.schemas.intervention import BoostRequest, BoostResponse
from app.services.intervention_engine import InterventionEngine

router = APIRouter(prefix="/api/v1/boost", tags=["boost"])


@router.post("/request", response_model=BoostResponse)
async def request_boost(
    data: BoostRequest,
    participant_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Request on-demand boost support.

    Presents a brief assessment (anxiety, sadness, location),
    then samples from the boost bandit and delivers intervention.
    Subject to 1-hour cooldown (separate from survey-triggered cooldown).
    """
    # Validate participant
    result = await db.execute(
        select(Participant).where(Participant.id == participant_id)
    )
    participant = result.scalar_one_or_none()
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")

    if participant.condition != "emi":
        raise HTTPException(status_code=403, detail="Boost only available for EMI condition")

    na_score = (data.anxiety + data.sadness) / 2.0

    # Get preferences
    pref_result = await db.execute(
        select(ParticipantPreference)
        .where(ParticipantPreference.participant_id == participant.id)
    )
    prefs = pref_result.scalar_one_or_none()
    wellness_prefs = prefs.preferred_activities if prefs else []

    engine = InterventionEngine(db)
    boost = await engine.handle_boost(
        participant_id=participant.id,
        na_score=na_score,
        context={
            "na_score": na_score,
            "anxiety": data.anxiety,
            "sadness": data.sadness,
            "location": data.location,
            "social_context": "unknown",
            "current_activity": "",
        },
        wellness_preferences=wellness_prefs,
    )

    if boost is None:
        raise HTTPException(
            status_code=429,
            detail="Boost is on cooldown. Please wait before requesting again.",
        )

    await db.commit()

    return BoostResponse(
        id=boost.id,
        boost_arm=boost.boost_arm,
        content_text=boost.content_text,
        trigger_na_score=boost.trigger_na_score,
        delivered_at=boost.delivered_at,
    )
