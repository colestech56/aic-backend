import uuid
from datetime import datetime

from sqlalchemy import Double, Float, ForeignKey, Integer, JSON
from sqlalchemy import TIMESTAMP, Uuid as UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class ThresholdState(Base):
    __tablename__ = "threshold_state"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    participant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, unique=True
    )
    current_threshold: Mapped[float] = mapped_column(Double, default=4.0)
    na_scores: Mapped[list] = mapped_column(JSON, default=list)
    n_surveys: Mapped[int] = mapped_column(Integer, default=0)
    individual_weight: Mapped[float] = mapped_column(Double, default=0.0)
    last_recalculated_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)
