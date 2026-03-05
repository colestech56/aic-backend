"""
Bayesian Shrinkage Estimator for personalized intervention trigger thresholds.

Uses population-level priors from prior EMA data (N=240) to initialize
thresholds on Day 1, then progressively shifts toward individual data
as each participant's survey responses accumulate.

Section 2 of the AIC spec.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import settings


@dataclass(frozen=True)
class PopulationPrior:
    """Population-level prior parameters for a diagnostic group."""

    na_mean: float
    na_sd: float
    threshold: float


# Table 1 values from the AIC spec (Section 2.1)
POPULATION_PRIORS: dict[str, PopulationPrior] = {
    "SZ": PopulationPrior(na_mean=2.70, na_sd=1.34, threshold=4.00),
    "BD": PopulationPrior(na_mean=2.82, na_sd=1.29, threshold=4.00),
}


class BayesianShrinkageEstimator:
    """
    Computes personalized NA trigger thresholds using a Bayesian
    shrinkage estimator that blends population priors with individual data.

    Properties (from Section 2.2):
    - Day 1, Survey 1 (n=0): threshold is entirely population prior (4.00)
    - By Day 3 (n≈9): ~43% individual, ~57% population
    - By Day 7 (n≈21): fully individualized
    - Smooth, monotonic transition with no discontinuity
    """

    def __init__(
        self,
        floor: float = settings.THRESHOLD_FLOOR,
        ceiling: float = settings.THRESHOLD_CEILING,
        n_target: int = settings.INDIVIDUALIZATION_TARGET_N,
        cv_check_min: int = settings.CV_CHECK_MIN_SURVEYS,
        cv_threshold: float = settings.CV_THRESHOLD,
        sd_multiplier: float = settings.SD_MULTIPLIER,
    ):
        self.floor = floor
        self.ceiling = ceiling
        self.n_target = n_target
        self.cv_check_min = cv_check_min
        self.cv_threshold = cv_threshold
        self.sd_multiplier = sd_multiplier

    def compute_threshold(
        self,
        diagnostic_group: str,
        na_scores: list[float],
    ) -> tuple[float, float]:
        """
        Compute the current trigger threshold for a participant.

        Args:
            diagnostic_group: 'SZ' or 'BD'
            na_scores: All NA scores observed so far for this participant.

        Returns:
            Tuple of (threshold, individual_weight).
            - threshold: The trigger threshold, clamped to [floor, ceiling].
            - individual_weight: w in [0, 1], how much the threshold is
              driven by individual vs population data.
        """
        prior = POPULATION_PRIORS[diagnostic_group]
        n = len(na_scores)

        # No data yet: pure population prior
        if n == 0:
            return (prior.threshold, 0.0)

        # Individual weight: linear ramp from 0 to 1 over n_target surveys
        w = min(1.0, n / self.n_target)

        scores = np.array(na_scores, dtype=np.float64)
        mean_na = float(np.mean(scores))
        sd_na = float(np.std(scores, ddof=1)) if n > 1 else 0.0

        # CV stability check (Section 2.3)
        # If CV > 0.50 after 9+ surveys, use median instead of mean
        if n >= self.cv_check_min and mean_na > 0:
            cv = sd_na / mean_na
            if cv > self.cv_threshold:
                center = float(np.median(scores))
            else:
                center = mean_na
        else:
            center = mean_na

        # Individual threshold: center + 1.0 * SD
        individual_threshold = center + self.sd_multiplier * sd_na

        # Blended threshold: weighted combination of individual and population
        blended = w * individual_threshold + (1 - w) * prior.threshold

        # Apply hard constraints (Section 2.3)
        clamped = max(self.floor, min(self.ceiling, blended))

        return (clamped, w)

    def get_population_prior(self, diagnostic_group: str) -> PopulationPrior:
        """Get the population prior for a diagnostic group."""
        return POPULATION_PRIORS[diagnostic_group]
