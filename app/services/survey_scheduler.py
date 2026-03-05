"""
Survey Scheduler for EMA, micro-survey, and end-of-day timing.

Generates stratified random EMA survey times within participant's waking hours,
with 2-hour minimum inter-survey intervals. Section 4.5 of the AIC spec.
"""

from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from app.config import settings


class SurveyScheduler:
    """
    Generates personalized survey schedules for each participant.

    EMA surveys: 3/day, stratified random within waking hours.
    Micro-surveys: 15 min after each intervention delivery.
    End-of-day: 30 min before sleep time.
    """

    NUM_EMA = settings.EMA_SURVEYS_PER_DAY
    MIN_GAP_MINUTES = settings.MIN_INTER_SURVEY_MINUTES
    EMA_WINDOW_MINUTES = settings.EMA_RESPONSE_WINDOW_MINUTES
    MICRO_DELAY_MINUTES = settings.MICRO_SURVEY_DELAY_MINUTES
    MICRO_WINDOW_MINUTES = settings.MICRO_SURVEY_WINDOW_MINUTES
    EOD_WINDOW_MINUTES = settings.EOD_RESPONSE_WINDOW_MINUTES

    def generate_daily_ema_schedule(
        self,
        wake_time: time,
        sleep_time: time,
        target_date: date,
        timezone: str,
    ) -> list[dict]:
        """
        Generate 3 stratified-random EMA times for a given day.

        Divides waking hours into 3 equal windows, picks a random
        time in each, then validates the 2-hour minimum gap.

        Returns list of dicts with 'scheduled_at' and 'window_closes_at'.
        """
        tz = ZoneInfo(timezone)
        wake_dt = datetime.combine(target_date, wake_time, tzinfo=tz)
        sleep_dt = datetime.combine(target_date, sleep_time, tzinfo=tz)

        # Handle case where sleep_time is before wake_time (e.g., night owl)
        if sleep_dt <= wake_dt:
            sleep_dt += timedelta(days=1)

        waking_minutes = (sleep_dt - wake_dt).total_seconds() / 60
        window_size = waking_minutes / self.NUM_EMA

        times = []
        for i in range(self.NUM_EMA):
            window_start = wake_dt + timedelta(minutes=i * window_size)
            window_end = wake_dt + timedelta(minutes=(i + 1) * window_size)

            # Add 15-min buffer from window edges
            buffer = timedelta(minutes=15)
            effective_start = window_start + buffer
            effective_end = window_end - buffer

            if effective_end <= effective_start:
                # Window too small for buffer, use midpoint
                rand_time = window_start + timedelta(minutes=window_size / 2)
            else:
                range_minutes = (effective_end - effective_start).total_seconds() / 60
                rand_time = effective_start + timedelta(
                    minutes=random.random() * range_minutes
                )

            times.append(rand_time)

        # Enforce minimum gap
        times = self._enforce_minimum_gap(times)

        return [
            {
                "scheduled_at": t,
                "window_closes_at": t + timedelta(minutes=self.EMA_WINDOW_MINUTES),
                "survey_type": "ema",
            }
            for t in times
        ]

    def schedule_micro_survey(
        self,
        intervention_delivered_at: datetime,
    ) -> dict:
        """Schedule micro-survey 15 minutes after intervention delivery."""
        scheduled = intervention_delivered_at + timedelta(minutes=self.MICRO_DELAY_MINUTES)
        return {
            "scheduled_at": scheduled,
            "window_closes_at": scheduled + timedelta(minutes=self.MICRO_WINDOW_MINUTES),
            "survey_type": "micro",
        }

    def schedule_end_of_day(
        self,
        sleep_time: time,
        target_date: date,
        timezone: str,
    ) -> dict:
        """Schedule end-of-day survey 30 min before sleep time."""
        tz = ZoneInfo(timezone)
        sleep_dt = datetime.combine(target_date, sleep_time, tzinfo=tz)
        scheduled = sleep_dt - timedelta(minutes=30)
        return {
            "scheduled_at": scheduled,
            "window_closes_at": scheduled + timedelta(minutes=self.EOD_WINDOW_MINUTES),
            "survey_type": "end_of_day",
        }

    def _enforce_minimum_gap(self, times: list[datetime]) -> list[datetime]:
        """
        Adjust survey times to ensure minimum 2-hour gap between any two.
        Shifts later surveys forward if they're too close to the previous one.
        """
        if len(times) <= 1:
            return times

        min_gap = timedelta(minutes=self.MIN_GAP_MINUTES)
        adjusted = [times[0]]

        for t in times[1:]:
            prev = adjusted[-1]
            if t - prev < min_gap:
                t = prev + min_gap
            adjusted.append(t)

        return adjusted
