"""
Reward Calculator for bandit posterior updates.

Computes combined reward from NA change (70%) and subjective helpfulness (30%).
Section 5.5 of the AIC spec.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardComponents:
    """Breakdown of the reward signal."""

    na_change_score: float
    helpfulness_score: float
    combined_reward: float
    normalized_change: float | None = None


class RewardCalculator:
    """
    Computes the combined reward for bandit updates.

    Primary reward (70%): NA change score
        Based on normalized change from pre- to post-intervention NA.
        normalized_change = (NA_pre - NA_post) / (NA_pre - 1)

    Secondary reward (30%): Subjective helpfulness
        Normalized from 1-5 scale to [0, 1].

    Combined: 0.70 * na_reward + 0.30 * helpfulness_reward
    """

    NA_WEIGHT: float = 0.70
    HELP_WEIGHT: float = 0.30

    def compute(
        self,
        na_pre: float,
        na_post: float | None,
        helpfulness: float | None,
    ) -> RewardComponents:
        """
        Compute the combined reward.

        Args:
            na_pre: Pre-intervention NA score (from triggering survey or boost).
            na_post: Post-intervention NA score (from micro-survey M1+M2 average).
                     None if micro-survey was not completed.
            helpfulness: Helpfulness rating 1-5 (micro-survey M3).
                         None if micro-survey was not completed.

        Returns:
            RewardComponents with all reward breakdowns.
        """
        normalized_change = None

        # Primary: NA change score
        if na_post is not None and na_pre > 1.0:
            normalized_change = (na_pre - na_post) / (na_pre - 1.0)

            if normalized_change >= 0.5:
                na_reward = 1.0
            elif normalized_change >= 0.25:
                na_reward = 0.75
            elif normalized_change >= 0.0:
                na_reward = 0.5
            else:
                na_reward = 0.2
        elif na_post is not None and na_pre <= 1.0:
            # Floor case: NA at minimum, shouldn't have triggered
            na_reward = 0.5
        else:
            # Missing micro-survey: neutral
            na_reward = 0.5

        # Secondary: Helpfulness (1-5 → 0-1)
        if helpfulness is not None:
            help_reward = (helpfulness - 1.0) / 4.0
        else:
            help_reward = 0.5  # neutral for missing

        combined = self.NA_WEIGHT * na_reward + self.HELP_WEIGHT * help_reward

        return RewardComponents(
            na_change_score=na_reward,
            helpfulness_score=help_reward,
            combined_reward=combined,
            normalized_change=normalized_change,
        )
