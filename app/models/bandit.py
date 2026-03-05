import uuid
from datetime import datetime

from sqlalchemy import Double, Integer, String, UniqueConstraint
from sqlalchemy import TIMESTAMP, Uuid as UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class BanditArm(Base):
    __tablename__ = "bandit_arms"
    __table_args__ = (
        UniqueConstraint("participant_id", "bandit_type", "arm_index"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    participant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    bandit_type: Mapped[str] = mapped_column(String(20), nullable=False)  # timing, intervention_type, boost
    arm_name: Mapped[str] = mapped_column(String(40), nullable=False)
    arm_index: Mapped[int] = mapped_column(Integer, nullable=False)
    alpha: Mapped[float] = mapped_column(Double, default=1.0)
    beta_param: Mapped[float] = mapped_column("beta", Double, default=1.0)
    times_selected: Mapped[int] = mapped_column(Integer, default=0)
    total_reward: Mapped[float] = mapped_column(Double, default=0.0)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)


class PopulationPrior(Base):
    __tablename__ = "population_priors"

    diagnostic_group: Mapped[str] = mapped_column(String(2), primary_key=True)
    na_mean: Mapped[float] = mapped_column(Double, nullable=False)
    na_sd: Mapped[float] = mapped_column(Double, nullable=False)
    threshold: Mapped[float] = mapped_column(Double, nullable=False)
