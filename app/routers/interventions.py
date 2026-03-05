"""Intervention state and tracking endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.intervention import Intervention
from app.schemas.intervention import InterventionResponse

router = APIRouter(prefix="/api/v1/interventions", tags=["interventions"])


@router.get("/current", response_model=InterventionResponse | None)
async def get_current_intervention(
    participant_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get the current pending or recently delivered intervention."""
    result = await db.execute(
        select(Intervention)
        .where(and_(
            Intervention.participant_id == participant_id,
            Intervention.status.in_(["pending", "delivered"]),
        ))
        .order_by(Intervention.created_at.desc())
        .limit(1)
    )
    intervention = result.scalar_one_or_none()
    if not intervention:
        return None

    # Check if pending intervention should be delivered now
    if (
        intervention.status == "pending"
        and intervention.scheduled_delivery_at
        and datetime.utcnow() >= intervention.scheduled_delivery_at
    ):
        intervention.status = "delivered"
        intervention.delivered_at = datetime.utcnow()
        await db.commit()

    return intervention


@router.post("/{intervention_id}/viewed")
async def mark_intervention_viewed(
    intervention_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Mark an intervention as viewed by the participant."""
    result = await db.execute(
        select(Intervention).where(Intervention.id == intervention_id)
    )
    intervention = result.scalar_one_or_none()
    if not intervention:
        raise HTTPException(status_code=404, detail="Intervention not found")

    intervention.status = "viewed"
    await db.commit()
    return {"status": "viewed"}
