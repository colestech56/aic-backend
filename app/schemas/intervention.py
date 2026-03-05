import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class InterventionResponse(BaseModel):
    id: uuid.UUID
    timing_arm: str
    intervention_type_arm: str
    content_text: str
    content_source: str
    status: str
    scheduled_delivery_at: datetime | None
    delivered_at: datetime | None
    trigger_na_score: float
    threshold_at_trigger: float
    reward_combined: float | None
    created_at: datetime

    model_config = {"from_attributes": True}


class BoostRequest(BaseModel):
    anxiety: int = Field(..., ge=1, le=7)
    sadness: int = Field(..., ge=1, le=7)
    location: str = Field(..., pattern="^(home|away|in_transit)$")


class BoostResponse(BaseModel):
    id: uuid.UUID
    boost_arm: str
    content_text: str
    trigger_na_score: float
    delivered_at: datetime

    model_config = {"from_attributes": True}


class BanditArmState(BaseModel):
    arm_name: str
    arm_index: int
    alpha: float
    beta: float
    times_selected: int
    total_reward: float
    mean_reward: float
    kl_divergence: float


class BanditStateResponse(BaseModel):
    participant_id: uuid.UUID
    bandit_type: str
    arms: list[BanditArmState]
    converged: bool


class ThresholdStateResponse(BaseModel):
    participant_id: uuid.UUID
    current_threshold: float
    individual_weight: float
    n_surveys: int
    diagnostic_group: str
    population_prior: float
