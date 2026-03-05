import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class EMAResponseData(BaseModel):
    """Regular EMA survey response."""
    anxiety: int = Field(..., ge=1, le=7)
    sadness: int = Field(..., ge=1, le=7)
    happiness: int = Field(..., ge=1, le=7)
    relaxation: int = Field(..., ge=1, le=7)
    # Psychotic symptoms (present/absent)
    hearing_voices: bool = False
    paranoia: bool = False
    special_messages: bool = False
    mind_reading: bool = False
    special_powers: bool = False
    # Context
    location: str = Field(..., pattern="^(home|away|in_transit)$")
    social_context: str = Field(..., pattern="^(alone|with_someone)$")
    current_activity: str = ""
    # Boost tracking (EMI group only)
    used_boost_since_last: bool | None = None


class MicroSurveyResponseData(BaseModel):
    """Micro-survey delivered 15 min post-intervention."""
    anxiety: int = Field(..., ge=1, le=7)
    sadness: int = Field(..., ge=1, le=7)
    helpfulness: int = Field(..., ge=1, le=5)


class BoostSurveyResponseData(BaseModel):
    """Brief assessment before boost delivery."""
    anxiety: int = Field(..., ge=1, le=7)
    sadness: int = Field(..., ge=1, le=7)
    location: str = Field(..., pattern="^(home|away|in_transit)$")


class EndOfDayResponseData(BaseModel):
    """End-of-day survey."""
    overall_day: int = Field(..., ge=1, le=7)
    na_interference: int = Field(..., ge=1, le=7)
    unusual_experiences: bool = False
    intervention_usefulness: int | None = Field(default=None, ge=1, le=5)


class SurveySubmission(BaseModel):
    response_data: dict


class SurveyScheduleResponse(BaseModel):
    id: uuid.UUID
    survey_type: str
    scheduled_at: datetime
    window_closes_at: datetime
    status: str
    linked_intervention_id: uuid.UUID | None

    model_config = {"from_attributes": True}


class SurveyResponseOut(BaseModel):
    id: uuid.UUID
    survey_type: str
    na_score: float | None
    submitted_at: datetime
    intervention_triggered: bool = False
    intervention_id: uuid.UUID | None = None

    model_config = {"from_attributes": True}
