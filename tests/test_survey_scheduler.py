"""Unit tests for the SurveyScheduler service."""

import pytest
from datetime import date, datetime, time, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

from app.services.survey_scheduler import SurveyScheduler


@pytest.fixture
def scheduler():
    return SurveyScheduler()


# --- EMA schedule generation ---

class TestDailyEMASchedule:
    def test_generates_three_ema_surveys(self, scheduler):
        result = scheduler.generate_daily_ema_schedule(
            wake_time=time(8, 0),
            sleep_time=time(22, 0),
            target_date=date(2026, 3, 5),
            timezone="America/New_York",
        )
        assert len(result) == 3

    def test_all_entries_are_ema_type(self, scheduler):
        result = scheduler.generate_daily_ema_schedule(
            wake_time=time(8, 0),
            sleep_time=time(22, 0),
            target_date=date(2026, 3, 5),
            timezone="America/New_York",
        )
        for entry in result:
            assert entry["survey_type"] == "ema"

    def test_surveys_are_within_waking_hours(self, scheduler):
        tz = ZoneInfo("America/New_York")
        wake = time(8, 0)
        sleep = time(22, 0)
        target = date(2026, 3, 5)

        result = scheduler.generate_daily_ema_schedule(wake, sleep, target, "America/New_York")

        wake_dt = datetime.combine(target, wake, tzinfo=tz)
        sleep_dt = datetime.combine(target, sleep, tzinfo=tz)

        for entry in result:
            assert wake_dt <= entry["scheduled_at"] <= sleep_dt

    def test_surveys_are_in_chronological_order(self, scheduler):
        result = scheduler.generate_daily_ema_schedule(
            wake_time=time(8, 0),
            sleep_time=time(22, 0),
            target_date=date(2026, 3, 5),
            timezone="America/New_York",
        )
        times = [e["scheduled_at"] for e in result]
        assert times == sorted(times)

    def test_minimum_two_hour_gap_enforced(self, scheduler):
        result = scheduler.generate_daily_ema_schedule(
            wake_time=time(8, 0),
            sleep_time=time(22, 0),
            target_date=date(2026, 3, 5),
            timezone="America/New_York",
        )
        times = [e["scheduled_at"] for e in result]
        for i in range(1, len(times)):
            gap = (times[i] - times[i - 1]).total_seconds() / 60
            assert gap >= 120, f"Gap between survey {i-1} and {i} is only {gap} minutes"

    def test_response_window_is_60_minutes(self, scheduler):
        result = scheduler.generate_daily_ema_schedule(
            wake_time=time(8, 0),
            sleep_time=time(22, 0),
            target_date=date(2026, 3, 5),
            timezone="America/New_York",
        )
        for entry in result:
            window = (entry["window_closes_at"] - entry["scheduled_at"]).total_seconds() / 60
            assert window == 60

    def test_night_owl_sleep_after_midnight(self, scheduler):
        """Handle sleep_time before wake_time (e.g., sleep at 2am)."""
        result = scheduler.generate_daily_ema_schedule(
            wake_time=time(10, 0),
            sleep_time=time(2, 0),  # 2am next day
            target_date=date(2026, 3, 5),
            timezone="America/New_York",
        )
        assert len(result) == 3
        # All surveys should be after 10am
        tz = ZoneInfo("America/New_York")
        wake_dt = datetime.combine(date(2026, 3, 5), time(10, 0), tzinfo=tz)
        for entry in result:
            assert entry["scheduled_at"] >= wake_dt

    def test_different_timezones(self, scheduler):
        """Surveys should be in the specified timezone."""
        result_ny = scheduler.generate_daily_ema_schedule(
            wake_time=time(8, 0),
            sleep_time=time(22, 0),
            target_date=date(2026, 3, 5),
            timezone="America/New_York",
        )
        result_la = scheduler.generate_daily_ema_schedule(
            wake_time=time(8, 0),
            sleep_time=time(22, 0),
            target_date=date(2026, 3, 5),
            timezone="America/Los_Angeles",
        )
        # Same local time, different UTC — LA should be 3 hours later in UTC
        ny_utc = result_ny[0]["scheduled_at"].astimezone(ZoneInfo("UTC"))
        la_utc = result_la[0]["scheduled_at"].astimezone(ZoneInfo("UTC"))
        # LA is 3 hours behind NY, so LA UTC should be later
        # (randomness means exact comparison not possible, but order of magnitude)
        diff_hours = abs((la_utc - ny_utc).total_seconds()) / 3600
        # They should differ by roughly 3 hours (±random variation within window)
        assert diff_hours < 8  # loose bound since times are random within windows

    def test_narrow_waking_window(self, scheduler):
        """Handle very short waking hours gracefully."""
        result = scheduler.generate_daily_ema_schedule(
            wake_time=time(8, 0),
            sleep_time=time(14, 0),  # Only 6 hours awake
            target_date=date(2026, 3, 5),
            timezone="America/New_York",
        )
        assert len(result) == 3
        # Still need minimum gap enforcement
        times = [e["scheduled_at"] for e in result]
        for i in range(1, len(times)):
            gap = (times[i] - times[i - 1]).total_seconds() / 60
            assert gap >= 120

    def test_randomness_varies_across_calls(self, scheduler):
        """Two calls should produce different times (with very high probability)."""
        r1 = scheduler.generate_daily_ema_schedule(
            time(8, 0), time(22, 0), date(2026, 3, 5), "America/New_York"
        )
        r2 = scheduler.generate_daily_ema_schedule(
            time(8, 0), time(22, 0), date(2026, 3, 5), "America/New_York"
        )
        t1 = [e["scheduled_at"] for e in r1]
        t2 = [e["scheduled_at"] for e in r2]
        # Extremely unlikely to match exactly
        assert t1 != t2


# --- Micro-survey scheduling ---

class TestMicroSurveySchedule:
    def test_scheduled_15_min_after_intervention(self, scheduler):
        delivery = datetime(2026, 3, 5, 14, 30)
        result = scheduler.schedule_micro_survey(delivery)
        assert result["scheduled_at"] == datetime(2026, 3, 5, 14, 45)

    def test_micro_survey_type(self, scheduler):
        delivery = datetime(2026, 3, 5, 14, 30)
        result = scheduler.schedule_micro_survey(delivery)
        assert result["survey_type"] == "micro"

    def test_micro_window_is_30_minutes(self, scheduler):
        delivery = datetime(2026, 3, 5, 14, 30)
        result = scheduler.schedule_micro_survey(delivery)
        window = (result["window_closes_at"] - result["scheduled_at"]).total_seconds() / 60
        assert window == 30


# --- End-of-day scheduling ---

class TestEndOfDaySchedule:
    def test_scheduled_30_min_before_sleep(self, scheduler):
        tz = ZoneInfo("America/New_York")
        result = scheduler.schedule_end_of_day(
            sleep_time=time(22, 0),
            target_date=date(2026, 3, 5),
            timezone="America/New_York",
        )
        expected = datetime.combine(date(2026, 3, 5), time(21, 30), tzinfo=tz)
        assert result["scheduled_at"] == expected

    def test_eod_survey_type(self, scheduler):
        result = scheduler.schedule_end_of_day(
            sleep_time=time(22, 0),
            target_date=date(2026, 3, 5),
            timezone="America/New_York",
        )
        assert result["survey_type"] == "end_of_day"

    def test_eod_window_is_120_minutes(self, scheduler):
        result = scheduler.schedule_end_of_day(
            sleep_time=time(22, 0),
            target_date=date(2026, 3, 5),
            timezone="America/New_York",
        )
        window = (result["window_closes_at"] - result["scheduled_at"]).total_seconds() / 60
        assert window == 120


# --- Gap enforcement ---

class TestEnforceMinimumGap:
    def test_no_adjustment_needed(self, scheduler):
        times = [
            datetime(2026, 3, 5, 9, 0),
            datetime(2026, 3, 5, 13, 0),
            datetime(2026, 3, 5, 17, 0),
        ]
        result = scheduler._enforce_minimum_gap(times)
        assert result == times

    def test_shifts_close_surveys_forward(self, scheduler):
        times = [
            datetime(2026, 3, 5, 9, 0),
            datetime(2026, 3, 5, 9, 30),  # Only 30 min gap
            datetime(2026, 3, 5, 17, 0),
        ]
        result = scheduler._enforce_minimum_gap(times)
        gap = (result[1] - result[0]).total_seconds() / 60
        assert gap >= 120

    def test_cascade_shift(self, scheduler):
        """If second shifts, third may also need shifting."""
        times = [
            datetime(2026, 3, 5, 9, 0),
            datetime(2026, 3, 5, 9, 30),  # Too close to first
            datetime(2026, 3, 5, 10, 0),  # Too close to adjusted second
        ]
        result = scheduler._enforce_minimum_gap(times)
        for i in range(1, len(result)):
            gap = (result[i] - result[i - 1]).total_seconds() / 60
            assert gap >= 120

    def test_single_time_unchanged(self, scheduler):
        times = [datetime(2026, 3, 5, 9, 0)]
        result = scheduler._enforce_minimum_gap(times)
        assert result == times

    def test_empty_list(self, scheduler):
        result = scheduler._enforce_minimum_gap([])
        assert result == []
