import uuid
from datetime import datetime

from sqlalchemy import Double, ForeignKey, String, JSON
from sqlalchemy import TIMESTAMP, Uuid as UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class SurveySchedule(Base):
    __tablename__ = "survey_schedules"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    participant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    survey_type: Mapped[str] = mapped_column(String(20), nullable=False)  # ema, micro, boost, end_of_day
    scheduled_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    window_closes_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="scheduled")  # scheduled, delivered, completed, expired, silenced
    delivered_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    linked_intervention_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)


class SurveyResponse(Base):
    __tablename__ = "survey_responses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    schedule_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    participant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    survey_type: Mapped[str] = mapped_column(String(20), nullable=False)
    response_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    na_score: Mapped[float | None] = mapped_column(Double, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)
