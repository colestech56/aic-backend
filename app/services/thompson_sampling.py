"""
Thompson Sampling Engine for multi-armed bandits.

Uses Beta-Bernoulli conjugate updates with KL divergence monitoring.
Three independent instances per participant: timing, intervention_type, boost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import digamma, gammaln
from scipy.stats import beta as beta_dist


@dataclass
class ArmState:
    """State of a single bandit arm."""

    arm_name: str
    arm_index: int
    alpha: float = 1.0
    beta: float = 1.0
    times_selected: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        if self.times_selected == 0:
            return 0.0
        return self.total_reward / self.times_selected

    @property
    def expected_value(self) -> float:
        """Expected value of the Beta distribution."""
        return self.alpha / (self.alpha + self.beta)


# Arm definitions from the AIC spec
TIMING_ARMS = [
    ArmState("immediate", 0),
    ArmState("5_min_delay", 1),
    ArmState("10_min_delay", 2),
    ArmState("next_survey_enhancement", 3),
]

INTERVENTION_TYPE_ARMS = [
    ArmState("sensory_focus", 0),
    ArmState("relaxation_breathing", 1),
    ArmState("activity_suggestion", 2),
    ArmState("social_outreach", 3),  # capped at 2/day
]

BOOST_ARMS = [
    ArmState("sensory_focus", 0),
    ArmState("relaxation_breathing", 1),
    ArmState("activity_suggestion", 2),
]


def create_fresh_arms(bandit_type: str) -> list[ArmState]:
    """Create a fresh set of arms with Beta(1,1) priors for a bandit type."""
    if bandit_type == "timing":
        return [ArmState(a.arm_name, a.arm_index) for a in TIMING_ARMS]
    elif bandit_type == "intervention_type":
        return [ArmState(a.arm_name, a.arm_index) for a in INTERVENTION_TYPE_ARMS]
    elif bandit_type == "boost":
        return [ArmState(a.arm_name, a.arm_index) for a in BOOST_ARMS]
    else:
        raise ValueError(f"Unknown bandit type: {bandit_type}")


class ThompsonSamplingEngine:
    """
    Manages a single multi-armed bandit for one participant.

    Uses Thompson sampling with Beta-Bernoulli conjugate updates.
    Samples from each arm's Beta posterior, selects the arm with
    the highest sample, and updates posteriors based on observed rewards.
    """

    def __init__(self, arms: list[ArmState]):
        self.arms = arms

    def select_arm(self, excluded_indices: list[int] | None = None) -> tuple[int, float]:
        """
        Sample from each arm's Beta posterior and return the arm with
        the highest sample.

        Args:
            excluded_indices: Arms to skip (e.g., social outreach at daily cap).

        Returns:
            Tuple of (selected_arm_index, sampled_value).
        """
        samples = []
        for arm in self.arms:
            if excluded_indices and arm.arm_index in excluded_indices:
                samples.append(-1.0)  # will never win
            else:
                samples.append(float(beta_dist.rvs(arm.alpha, arm.beta)))
        selected = int(np.argmax(samples))
        return selected, samples[selected]

    def update(self, arm_index: int, reward: float) -> None:
        """
        Beta-Bernoulli conjugate update.

        For continuous rewards in [0, 1], this treats the reward as
        a fractional success count.

        Args:
            arm_index: Index of the arm to update.
            reward: Reward value in [0, 1].
        """
        arm = self.arms[arm_index]
        arm.alpha += reward
        arm.beta += (1.0 - reward)
        arm.times_selected += 1
        arm.total_reward += reward

    def kl_divergence_from_uniform(self) -> dict[str, float]:
        """
        Compute KL(current_posterior || Beta(1,1)) for each arm.

        Used for convergence monitoring per Section 5.7 of the AIC spec.
        An arm is considered "learned" when KL > 0.10 nats.
        """
        results = {}
        for arm in self.arms:
            kl = kl_beta(arm.alpha, arm.beta, 1.0, 1.0)
            results[arm.arm_name] = kl
        return results

    def has_converged(self, threshold: float = 0.10) -> bool:
        """
        Check if at least one arm has KL divergence > threshold.

        Per Section 5.7: "A participant's bandit is considered 'converged'
        when at least one arm per bandit system exceeds this threshold."
        """
        kl_values = self.kl_divergence_from_uniform()
        return any(v > threshold for v in kl_values.values())

    def get_arm_by_index(self, index: int) -> ArmState:
        """Get an arm by its index."""
        for arm in self.arms:
            if arm.arm_index == index:
                return arm
        raise ValueError(f"No arm with index {index}")

    def get_arm_by_name(self, name: str) -> ArmState:
        """Get an arm by its name."""
        for arm in self.arms:
            if arm.arm_name == name:
                return arm
        raise ValueError(f"No arm with name {name}")


def kl_beta(a1: float, b1: float, a2: float, b2: float) -> float:
    """
    Compute KL divergence KL(Beta(a1,b1) || Beta(a2,b2)) using closed-form.

    Formula from Section 5.7:
    KL = ln(B(a2,b2)/B(a1,b1)) + (a1-a2)*psi(a1) + (b1-b2)*psi(b1)
         - (a1+b1-a2-b2)*psi(a1+b1)
    """
    return (
        gammaln(a2) + gammaln(b2) - gammaln(a2 + b2)
        - gammaln(a1) - gammaln(b1) + gammaln(a1 + b1)
        + (a1 - a2) * digamma(a1)
        + (b1 - b2) * digamma(b1)
        + (a2 - a1 + b2 - b1) * digamma(a1 + b1)
    )
