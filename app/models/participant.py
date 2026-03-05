import uuid
from datetime import datetime, time

from sqlalchemy import Boolean, String, Time, UniqueConstraint, JSON
from sqlalchemy import TIMESTAMP, Uuid as UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class Participant(Base):
    __tablename__ = "participants"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_id: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    diagnostic_group: Mapped[str] = mapped_column(String(2), nullable=False)  # SZ or BD
    condition: Mapped[str] = mapped_column(String(10), nullable=False, default="emi")  # emi or control
    enrolled_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    wake_time: Mapped[time] = mapped_column(Time, default=time(8, 0))
    sleep_time: Mapped[time] = mapped_column(Time, default=time(22, 0))
    timezone: Mapped[str] = mapped_column(String(50), default="America/New_York")
    silenced_until: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)


class ParticipantPreference(Base):
    __tablename__ = "participant_preferences"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    participant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, unique=True
    )
    preferred_activities: Mapped[list] = mapped_column(JSON, default=list)
    onboarding_data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)
