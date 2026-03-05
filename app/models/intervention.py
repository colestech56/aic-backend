import uuid
from datetime import datetime

from sqlalchemy import Boolean, Double, Integer, String, Text
from sqlalchemy import TIMESTAMP, Uuid as UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class Intervention(Base):
    __tablename__ = "interventions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    participant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    trigger_survey_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    trigger_na_score: Mapped[float] = mapped_column(Double, nullable=False)
    threshold_at_trigger: Mapped[float] = mapped_column(Double, nullable=False)
    # Bandit selections
    timing_arm: Mapped[str] = mapped_column(String(40), nullable=False)
    intervention_type_arm: Mapped[str] = mapped_column(String(40), nullable=False)
    timing_sample: Mapped[float] = mapped_column(Double, nullable=False)
    type_sample: Mapped[float] = mapped_column(Double, nullable=False)
    # Content
    content_text: Mapped[str] = mapped_column(Text, nullable=False)
    content_source: Mapped[str] = mapped_column(String(20), nullable=False)  # llm or static_fallback
    llm_raw_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    screening_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    screening_retries: Mapped[int] = mapped_column(Integer, default=0)
    cosine_sim_max: Mapped[float | None] = mapped_column(Double, nullable=True)
    # Delivery
    scheduled_delivery_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    delivered_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, delivered, viewed, expired
    # Reward (filled after micro-survey)
    reward_na_change: Mapped[float | None] = mapped_column(Double, nullable=True)
    reward_helpfulness: Mapped[float | None] = mapped_column(Double, nullable=True)
    reward_combined: Mapped[float | None] = mapped_column(Double, nullable=True)
    reward_calculated_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)


class BoostIntervention(Base):
    __tablename__ = "boost_interventions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    participant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    boost_arm: Mapped[str] = mapped_column(String(40), nullable=False)
    boost_sample: Mapped[float] = mapped_column(Double, nullable=False)
    content_text: Mapped[str] = mapped_column(Text, nullable=False)
    content_source: Mapped[str] = mapped_column(String(20), nullable=False)
    trigger_na_score: Mapped[float | None] = mapped_column(Double, nullable=True)
    delivered_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    # Reward
    reward_na_change: Mapped[float | None] = mapped_column(Double, nullable=True)
    reward_helpfulness: Mapped[float | None] = mapped_column(Double, nullable=True)
    reward_combined: Mapped[float | None] = mapped_column(Double, nullable=True)
    reward_calculated_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)


class DailyCounter(Base):
    __tablename__ = "daily_counters"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    participant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    counter_date: Mapped[str] = mapped_column(String(10), nullable=False)  # YYYY-MM-DD
    social_outreach_count: Mapped[int] = mapped_column(Integer, default=0)
    intervention_count: Mapped[int] = mapped_column(Integer, default=0)
