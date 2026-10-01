"""
did.py — estimate the markdown's effect, and test whether the estimator deserves to be believed.

WHY THIS FILE EXISTS
--------------------
The markdown months in `holding-360` are a natural experiment: four SKUs carry a clearance discount
during March, September and November, and forty SKUs never carry one. That is a difference-in-
differences (DiD) setting: compare how treated SKUs changed against how control SKUs changed over
the same window, so that anything affecting both groups is subtracted out.

But a DiD estimate is only as good as its **parallel-trends assumption** — the claim that, absent the
treatment, the two groups would have moved together. This module does not assume it. It does three
things a defensible causal analysis owes a reader:

1. **Estimates the effect** and compares it against the truth, which is knowable here because the
   world is synthetic. An estimator that cannot recover a known answer in a controlled setting has
   no business being trusted on an unknown one.
2. **Decomposes the control group's movement** into price/cost changes versus product-mix changes.
   If the movement is pure mix, then DiD is subtracting a composition artefact rather than a common
   shock — which is worth knowing, because it means the sophistication buys almost nothing.
3. **Runs a placebo test**: apply the same DiD machinery to a fake treatment inside months where no
   markdown ever happened. A well-behaved design returns roughly zero. Whatever it returns instead
   is the noise floor against which the real estimate must be judged.

WHAT A READER NEEDS TO KNOW FIRST
---------------------------------
* `unit_margin` is margin per unit in MXN — the outcome.
* "Naive" means the treated group's own before/after change, with no control group.
* "DiD" means the naive change minus the control group's change over the same period.

A NOTE ON INFERENCE
-------------------
There are only **four treated SKUs**. Treating 27 treated invoice lines as 27 independent
observations would understate the uncertainty badly, because lines of the same SKU are obviously
alike. The confidence interval here is therefore a **cluster bootstrap over SKUs**: whole SKUs are
resampled with replacement, so the effective sample size the interval reflects is 4 and 40, not 27
and 3,911. This is the honest interval, and it is much wider than the naive one.
"""

from __future__ import annotations

# dataclasses: typed result containers, so fields cannot be mixed up at the call site.
from dataclasses import dataclass, field

# numpy: the bootstrap and the arithmetic.
import numpy as np
# pandas: grouping and reshaping the panel.
import pandas as pd
# scipy.stats: Welch's t-test.
from scipy import stats


@dataclass
class DidEstimate:
    """The four cell means and the two estimates built from them.

    Attributes
    ----------
    treated_change:   treated group's before/after change (the "naive" estimate).
    control_change:   control group's change over the same period.
    did:              `treated_change - control_change`.
    cell_means:       the four means, keyed `(is_treated_sku, is_treated_period)`.
    counts:           how many lines sit in each cell.
    """

    treated_change: float
    control_change: float
    did: float
    cell_means: dict
    counts: dict


def _cell_means(frame: pd.DataFrame) -> tuple[dict, dict]:
    """Return the mean outcome and the count for each cell of the 2x2 grid."""
    grouped = frame.groupby(["treated_sku", "treated_period"])["unit_margin"]
    means = grouped.mean().to_dict()
    counts = grouped.size().to_dict()
    return means, counts


def estimate_did(frame: pd.DataFrame, outcome: str = "unit_margin") -> DidEstimate:
    """Run the 2x2 difference-in-differences on the panel.

    Parameters
    ----------
    frame:
        The panel from `panel.py`, with `treated_sku` and `treated_period` already set.
    outcome:
        The column to difference. Defaults to per-unit margin.

    Returns
    -------
    DidEstimate

    Raises
    ------
    ValueError
        If any cell of the 2x2 grid is empty. A missing cell means the comparison is undefined, and
        silently returning a number built from three cells would be worse than failing.
    """
    work = frame.rename(columns={outcome: "unit_margin"}) if outcome != "unit_margin" else frame
    means, counts = _cell_means(work)

    # The four cells a 2x2 DiD is built from.
    try:
        treated_in = means[(True, True)]
        treated_out = means[(True, False)]
        control_in = means[(False, True)]
        control_out = means[(False, False)]
    except KeyError as exc:
        raise ValueError(
            f"the 2x2 grid is incomplete — no observations for cell {exc}. "
            "A DiD needs all four cells."
        ) from exc

    treated_change = treated_in - treated_out
    control_change = control_in - control_out

    return DidEstimate(
        treated_change=float(treated_change),
        control_change=float(control_change),
        did=float(treated_change - control_change),
        cell_means={k: float(v) for k, v in means.items()},
        counts={k: int(v) for k, v in counts.items()},
    )


def welch_test(frame: pd.DataFrame, outcome: str = "unit_margin") -> tuple[float, float]:
    """Compare treated lines inside markdown months against treated lines outside them.

    This is the simplest possible test and it is deliberately included: when the treatment is large
    and mechanical, the simplest test is not just adequate, it is the one whose assumptions a
    reviewer can check in one line.

    Returns
    -------
    tuple[float, float]
        The t statistic and the two-sided p-value.
    """
    treated = frame[frame["treated_sku"]]
    inside = treated.loc[treated["treated_period"], outcome]
    outside = treated.loc[~treated["treated_period"], outcome]
    # equal_var=False -> Welch's t-test, which does not assume the two groups share a variance.
    stat, p_value = stats.ttest_ind(inside, outside, equal_var=False)
    return float(stat), float(p_value)


def mix_decomposition(control: pd.DataFrame) -> pd.DataFrame:
    """Split the control group's monthly margin movement into mix and price/cost components.

    HOW THE DECOMPOSITION WORKS
    ---------------------------
    Two monthly series are computed for the control group:

    * **Raw mean** — the plain average unit margin of every control line sold that month.
    * **Fixed-weight mean** — the average of each SKU's own monthly mean margin, weighted by that
      SKU's overall share of control lines, so the weights do not change from month to month.

    The difference between the two is the **mix effect**: what the raw series moves because a
    different assortment of products sold that month, holding nothing else constant.

    If the fixed-weight series is flat while the raw series swings, then the movement was entirely
    composition — and there was no price or cost shock for DiD to difference away. That is a finding
    about the design, not a technicality: it bounds what the second difference can possibly buy.

    Returns
    -------
    pandas.DataFrame
        Indexed by month with `raw_mean`, `fixed_weight_mean` and `mix_gap`.
    """
    raw = control.groupby("month")["unit_margin"].mean()

    # Each SKU's own monthly mean, as a month x SKU matrix.
    by_sku = control.groupby(["item_id", "month"])["unit_margin"].mean().unstack()
    # Fixed weights: each SKU's share of all control lines, in the panel overall.
    weights = control["item_id"].value_counts(normalize=True)

    # Recombine with fixed weights. Missing SKU-months are skipped rather than treated as zero, and
    # the weights are renormalised over whatever is present so a month with fewer SKUs is still on
    # the same scale.
    present = by_sku.notna().astype(float)
    numerator = (by_sku.fillna(0.0).T * weights).sum(axis=1)
    denominator = (present.T * weights).sum(axis=1).replace(0.0, np.nan)

    out = pd.DataFrame({
        "raw_mean": raw,
        "fixed_weight_mean": numerator / denominator,
    })
    out["mix_gap"] = out["raw_mean"] - out["fixed_weight_mean"]
    return out


def placebo_did(frame: pd.DataFrame, outcome: str = "unit_margin") -> DidEstimate:
    """Run the same DiD on a fake treatment, inside months where no markdown ever happens.

    WHAT THIS TESTS
    ---------------
    Restrict the data to non-markdown months, then declare every OTHER month a fake "treated"
    period. No treatment exists, so a well-behaved design returns approximately zero. What it
    actually returns is the **noise floor**: the size of estimate this design can produce from
    composition alone.

    If the placebo estimate is a meaningful share of the real estimate, the real estimate should not
    be read as a treatment effect, because the same machinery manufactures comparable numbers out of
    nothing.

    Returns
    -------
    DidEstimate
        The placebo estimate (its `did` field is the number that matters).
    """
    # Keep only months where the markdown is inactive — no real treatment can leak in.
    quiet = frame[~frame["treated_period"]].copy()
    # A fake period: odd calendar months. Arbitrary by construction, which is the point.
    quiet["treated_period"] = quiet["month_number"] % 2 == 1
    return estimate_did(quiet, outcome=outcome)


def cluster_bootstrap_did(
    frame: pd.DataFrame,
    outcome: str = "unit_margin",
    n_boot: int = 2_000,
    seed: int = 11,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the DiD estimate, resampling whole SKUs.

    WHY CLUSTER BY SKU
    ------------------
    Lines belonging to the same SKU are not independent observations — they share a price, a cost
    and a customer base. Treating them as independent would produce a confidently narrow interval
    around a number supported by only four treated SKUs. Resampling SKUs with replacement keeps the
    dependence intact and reflects the real amount of independent information in the data.

    Returns
    -------
    tuple[float, float, float]
        `(point_estimate, ci_low, ci_high)` at 95%, from the percentile method.
    """
    rng = np.random.default_rng(seed)
    work = frame.rename(columns={outcome: "unit_margin"}) if outcome != "unit_margin" else frame

    treated_skus = work.loc[work["treated_sku"], "item_id"].unique()
    control_skus = work.loc[~work["treated_sku"], "item_id"].unique()

    treated_by_sku = {s: g for s, g in work[work["treated_sku"]].groupby("item_id")}
    control_by_sku = {s: g for s, g in work[~work["treated_sku"]].groupby("item_id")}

    estimates = np.empty(n_boot)
    for i in range(n_boot):
        # Draw whole SKUs, with replacement — a treated SKU may appear twice, another may not appear.
        t_pick = rng.choice(treated_skus, size=len(treated_skus), replace=True)
        c_pick = rng.choice(control_skus, size=len(control_skus), replace=True)
        sample = pd.concat(
            [*[treated_by_sku[s] for s in t_pick], *[control_by_sku[s] for s in c_pick]],
            ignore_index=True,
        )
        try:
            estimates[i] = estimate_did(sample).did
        except ValueError:
            # A resample can miss a cell of the 2x2 grid entirely. Recording NaN and dropping it is
            # honest; inventing a zero would bias the interval toward the null.
            estimates[i] = np.nan

    valid = estimates[~np.isnan(estimates)]
    if valid.size == 0:
        return float("nan"), float("nan"), float("nan")

    low, high = np.percentile(valid, [2.5, 97.5])
    return float(estimate_did(work).did), float(low), float(high)
