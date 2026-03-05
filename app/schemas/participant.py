import uuid
from datetime import datetime, time

from pydantic import BaseModel, Field


class ParticipantCreate(BaseModel):
    study_id: str = Field(..., max_length=20, examples=["AIC-001"])
    diagnostic_group: str = Field(..., pattern="^(SZ|BD)$")
    condition: str = Field(default="emi", pattern="^(emi|control)$")
    wake_time: time = Field(default=time(8, 0))
    sleep_time: time = Field(default=time(22, 0))
    timezone: str = Field(default="America/New_York")
    preferred_activities: list[str] = Field(default=[])


class ParticipantUpdate(BaseModel):
    wake_time: time | None = None
    sleep_time: time | None = None
    timezone: str | None = None
    active: bool | None = None


class ParticipantResponse(BaseModel):
    id: uuid.UUID
    study_id: str
    diagnostic_group: str
    condition: str
    enrolled_at: datetime
    active: bool
    wake_time: time
    sleep_time: time
    timezone: str
    silenced_until: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}
