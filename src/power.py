"""
power.py — how many observations an experiment needs, and a check that the formula is right.

WHY THIS FILE EXISTS
--------------------
The screening question that cost the user a job asked for "power/sample-size calculations". That is
a concrete, checkable skill: given an effect you care about, a significance level and a target
power, produce the number of observations required. This module produces it two independent ways:

1. **Closed form** — the standard two-sample formula. Fast, and the one an interviewer expects.
2. **Simulation** — generate thousands of experiments under the assumed truth and measure how often
   the test rejects. This is the check: if the closed form were wrong (a mis-signed term, a
   one-sided/two-sided slip), the two would disagree, and the disagreement would be visible.

A FORMULA A READER NEEDS, EXPLAINED
-----------------------------------
    n per arm = 2 * (z_(1-alpha/2) + z_(1-beta))^2 / (delta / sigma)^2

* `delta / sigma` is the effect size: the change you want to detect, measured in standard
  deviations. It is unitless on purpose — the same arithmetic works for pesos, units or percentages.
* `z_(1-alpha/2)` = 1.96 at alpha = 5%: we allow a 5% chance of calling a difference that is not
  there (a false alarm).
* `z_(1-beta)` = 0.84 at 80% power: we allow a 20% chance of missing a difference that IS there.

The formula says the requirement grows with the SQUARE of the effect size. Halving the effect you
want to detect quadruples the sample. That single fact drives most of this project's conclusions.
"""

from __future__ import annotations

# dataclasses: a small typed container for the result, so callers cannot mix up the fields.
from dataclasses import dataclass

# numpy: the simulation and the normal quantile.
import numpy as np
# scipy.stats: the normal distribution's quantile function (the z values above).
from scipy import stats


@dataclass
class PowerResult:
    """What a sample-size calculation returns.

    Attributes
    ----------
    effect:      the change you want to detect, in the outcome's own units.
    sd:          the standard deviation of the outcome (the noise).
    effect_size: `effect / sd`, the unitless version.
    n_per_arm:   observations required in EACH arm.
    alpha:       the false-alarm rate the calculation used.
    power:       the probability of detecting the effect, if it is real.
    """

    effect: float
    sd: float
    effect_size: float
    n_per_arm: int
    alpha: float
    power: float


def required_sample_size(
    effect: float,
    sd: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> PowerResult:
    """Closed-form sample size for a two-arm comparison of means.

    Parameters
    ----------
    effect:
        The smallest change worth detecting. Use a real one: the effect that would change a
        decision. Detecting a 1% change nobody would act on is a waste of the sample.
    sd:
        Standard deviation of the outcome. This is the number people guess and should not — it is
        measurable from any existing data, which is why `panel.py` exists.
    alpha:
        Two-sided false-positive rate. 0.05 is conventional and is what "significant at 5%" means.
    power:
        Probability of detecting a real effect of size `effect`. 0.80 is conventional.

    Returns
    -------
    PowerResult
        Including `n_per_arm`, which is rounded UP to a whole number of observations.

    Raises
    ------
    ValueError
        If `sd` is not positive or `effect` is zero — both would make the formula divide by zero or
        return infinity, and an infinite sample requirement should be an error, not a number.
    """
    if sd <= 0:
        raise ValueError("sd must be positive; a zero-variance outcome needs no sample at all")
    if effect == 0:
        raise ValueError("effect must be non-zero; detecting nothing is not a sample-size question")

    # z for the false-alarm rate: 1.96 at alpha=0.05 (two-sided).
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    # z for the power: 0.84 at power=0.80.
    z_power = stats.norm.ppf(power)

    effect_size = abs(effect) / sd
    # ceil, not round: you cannot run 693.6 observations, and rounding down would quietly miss the
    # power target the caller asked for.
    n_per_arm = int(np.ceil(2 * (z_alpha + z_power) ** 2 / effect_size**2))

    return PowerResult(
        effect=effect, sd=sd, effect_size=effect_size,
        n_per_arm=n_per_arm, alpha=alpha, power=power,
    )


def simulate_power(
    effect: float,
    sd: float,
    n_per_arm: int,
    alpha: float = 0.05,
    n_trials: int = 4_000,
    seed: int = 7,
) -> float:
    """Measure the true detection rate by running many experiments, to check the formula.

    Each trial draws two samples from the same distribution, adds `effect` to one of them, runs a
    two-sample t-test, and records whether it rejected. The share that rejected IS the empirical
    power — no formula involved, which is exactly why it is a useful check.

    The seed is fixed so the result is reproducible; a power number that changes between runs would
    be useless as a test target.
    """
    rng = np.random.default_rng(seed)
    rejections = 0

    for _ in range(n_trials):
        # Control arm: noise only. Treatment arm: the same noise plus the effect.
        control = rng.normal(0.0, sd, n_per_arm)
        treatment = rng.normal(effect, sd, n_per_arm)
        # Welch's t-test: it does not assume the two arms have equal variance, which is the safe
        # default and the one to reach for unless there is a reason not to.
        _, p_value = stats.ttest_ind(treatment, control, equal_var=False)
        if p_value < alpha:
            rejections += 1

    return rejections / n_trials
