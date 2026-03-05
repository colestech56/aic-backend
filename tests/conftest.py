"""Shared fixtures for AIC backend tests."""

import uuid
from datetime import datetime, time, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest


def make_uuid():
    return uuid.uuid4()


@pytest.fixture
def participant_id():
    return make_uuid()


@pytest.fixture
def mock_participant(participant_id):
    """A mock Participant object."""
    p = MagicMock()
    p.id = participant_id
    p.study_id = "AIC-TEST-001"
    p.diagnostic_group = "SZ"
    p.condition = "emi"
    p.enrolled_at = datetime.utcnow()
    p.active = True
    p.wake_time = time(8, 0)
    p.sleep_time = time(22, 0)
    p.timezone = "America/New_York"
    p.silenced_until = None
    p.created_at = datetime.utcnow()
    p.updated_at = datetime.utcnow()
    return p


@pytest.fixture
def mock_survey_schedule(participant_id):
    """A mock SurveySchedule for EMA."""
    s = MagicMock()
    s.id = make_uuid()
    s.participant_id = participant_id
    s.survey_type = "ema"
    s.scheduled_at = datetime.utcnow() - timedelta(minutes=5)
    s.window_closes_at = datetime.utcnow() + timedelta(minutes=55)
    s.status = "scheduled"
    s.delivered_at = None
    s.completed_at = None
    s.linked_intervention_id = None
    s.created_at = datetime.utcnow()
    return s
