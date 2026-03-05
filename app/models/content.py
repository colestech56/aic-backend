import uuid
from datetime import datetime

from sqlalchemy import Boolean, String, Text
from sqlalchemy import TIMESTAMP, Uuid as UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class StaticFallback(Base):
    __tablename__ = "static_fallbacks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    intervention_type: Mapped[str] = mapped_column(String(40), nullable=False)
    content_text: Mapped[str] = mapped_column(Text, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class ContentHistory(Base):
    __tablename__ = "content_history"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    participant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    intervention_type: Mapped[str] = mapped_column(String(40), nullable=False)
    content_text: Mapped[str] = mapped_column(Text, nullable=False)
    delivered_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), default=datetime.utcnow)
