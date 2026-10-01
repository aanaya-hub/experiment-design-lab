"""
main.py — run the whole analysis and write `reports/results.json`.

WHY THIS FILE EXISTS
--------------------
Every number that appears in the README, the notebook or a conversation is produced here, once. If
a figure in the write-up cannot be traced to this file, it should not be quoted. That rule exists
because the alternative — numbers typed by hand into prose — is how a portfolio project ends up
claiming something it cannot reproduce.

HOW TO RUN
----------
    python src/main.py

It prints a human-readable report and writes `reports/results.json`.

THE FOUR QUESTIONS IT ANSWERS
-----------------------------
1. **Is the markdown where the design says it is?** (integrity — a mislabelled treatment invalidates
   everything downstream)
2. **What did the markdown do to margin?** (the estimate, the naive comparison, and the known truth)
3. **Does the design deserve to be believed?** (parallel trends, mix decomposition, placebo)
4. **What could this design ever have detected?** (power, at the effect sizes that would matter)
"""

from __future__ import annotations

# json: the results payload, so other artifacts read numbers instead of re-deriving them.
import json
# pathlib: the output path.
from pathlib import Path

# numpy / pandas: summary statistics on the panel.
import numpy as np
import pandas as pd

# The project's own modules.
from panel import TREATED_MONTHS, TREATED_SKUS, load_panel
from did import cluster_bootstrap_did, estimate_did, mix_decomposition, placebo_did, welch_test
from power import required_sample_size, simulate_power

# --- Output location ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "reports"

#: Fixed so the analysis is reproducible; a power number that changes between runs is not testable.
SEED = 7


def main() -> dict:
    """Run the analysis, print it, write it, and return the payload."""
    panel = load_panel()
    frame = panel.frame

    print("=" * 78)
    print("EXPERIMENT DESIGN LAB — the markdown months as a natural experiment")
    print("=" * 78)
    print("! Synthetic data from the sibling project holding-360. No real business is described.")

    # ---- 1. Panel integrity ------------------------------------------------------------------
    # The treatment must be exactly where the design says. Zero discounted lines outside the treated
    # cells is not a nice-to-have — it is the precondition for the estimate meaning anything.
    discounted = frame[frame["discounted"]]
    mislabelled = int((~discounted["treated_sku"] | ~discounted["treated_period"]).sum())

    print()
    print("-" * 78)
    print("1. INTEGRITY — is the treatment where the design says it is?")
    print("-" * 78)
    print(f"  invoice lines (BTM customer invoices) : {len(frame):,}")
    print(f"  months                                : {frame['month'].nunique()} "
          f"({frame['month'].min()} -> {frame['month'].max()})")
    print(f"  treated SKUs                          : {len(TREATED_SKUS)} {list(TREATED_SKUS)}")
    print(f"  control SKUs                          : {frame.loc[~frame['treated_sku'], 'item_id'].nunique()}")
    print(f"  treated months                        : {list(TREATED_MONTHS)}")
    print(f"  discounted lines                      : {len(discounted)}")
    print(f"  discounted lines OUTSIDE treated cells: {mislabelled}"
          f"   {'OK' if mislabelled == 0 else '! PROBLEM'}")

    grid = frame.groupby(["treated_sku", "treated_period"]).size().unstack(fill_value=0)
    print("  lines per cell of the 2x2 grid:")
    for row_label, row in grid.iterrows():
        who = "treated SKUs" if row_label else "control SKUs"
        print(f"    {who:<14} normal {row[False]:>5,}   markdown {row[True]:>5,}")

    # ---- 2. The estimate ---------------------------------------------------------------------
    est = estimate_did(frame)
    truth = panel.truth_att
    t_stat, p_value = welch_test(frame)

    naive_error = est.treated_change - truth
    did_error = est.did - truth

    print()
    print("-" * 78)
    print("2. THE EFFECT ON UNIT MARGIN")
    print("-" * 78)
    print(f"  treated SKUs, normal months    : {est.cell_means[(True, False)]:>12,.2f} MXN")
    print(f"  treated SKUs, markdown months  : {est.cell_means[(True, True)]:>12,.2f} MXN")
    print(f"  control SKUs, normal months    : {est.cell_means[(False, False)]:>12,.2f} MXN")
    print(f"  control SKUs, markdown months  : {est.cell_means[(False, True)]:>12,.2f} MXN")
    print()
    print(f"  naive own-change (treated)     : {est.treated_change:>12,.2f}   "
          f"error vs truth {naive_error:>8,.2f} ({abs(naive_error / truth):>5.1%})")
    print(f"  control change over same window: {est.control_change:>12,.2f}")
    print(f"  DiD estimate                   : {est.did:>12,.2f}   "
          f"error vs truth {did_error:>8,.2f} ({abs(did_error / truth):>5.1%})")
    print(f"  KNOWN TRUE EFFECT (ATT)        : {truth:>12,.2f}")
    print()
    print("  ! The truth is knowable because the world is synthetic: the markdown is a mechanical")
    print("  ! price cut, so its effect on margin is arithmetic, not behavioural.")
    print(f"  Welch t on treated lines       : t = {t_stat:,.2f},  p = {p_value:.3g}")

    point, ci_low, ci_high = cluster_bootstrap_did(frame, n_boot=2_000, seed=11)
    print(f"  cluster bootstrap 95% CI (over SKUs): [{ci_low:,.2f}, {ci_high:,.2f}]")
    print(f"    -> the interval covers the truth: {ci_low <= truth <= ci_high}")

    # ---- 3. Does the design deserve to be believed? ------------------------------------------
    mix = mix_decomposition(frame[~frame["treated_sku"]])
    placebo = placebo_did(frame)

    print()
    print("-" * 78)
    print("3. PARALLEL TRENDS — what is the second difference actually removing?")
    print("-" * 78)
    print("  control group, monthly mean unit margin:")
    print(f"    raw mean              : sd {mix['raw_mean'].std():>9,.2f}   "
          f"range {mix['raw_mean'].max() - mix['raw_mean'].min():>9,.2f}")
    print(f"    fixed-weight (no mix) : sd {mix['fixed_weight_mean'].std():>9,.2f}   "
          f"range {mix['fixed_weight_mean'].max() - mix['fixed_weight_mean'].min():>9,.2f}")
    print(f"    mix gap (raw - fixed) : sd {mix['mix_gap'].std():>9,.2f}")
    print()
    print("  ! A fixed-weight series that does not move means the control group's month-to-month")
    print("  ! swing is ENTIRELY product mix. There is no time-varying cost or price shock for DiD to")
    print("  ! remove — so the second difference corrects a composition artefact, not a confounder.")
    print(f"  placebo DiD on fake treatment in quiet months: {placebo.did:>12,.2f}")
    print(f"    -> placebo / real effect ratio: {abs(placebo.did / est.did):>5.1%}")

    # ---- 4. Power ----------------------------------------------------------------------------
    # (a) The retrospective question: the effect already happened. How big is the treated sample?
    treated_lines = int(est.counts[(True, True)])
    # (b) The prospective question: what would a real test need?
    treated_margin = frame.loc[frame["treated_sku"] & ~frame["treated_period"], "unit_margin"]
    base_mean, base_sd = float(treated_margin.mean()), float(treated_margin.std())

    margin_power = {}
    for pct in (0.05, 0.10, 0.15):
        res = required_sample_size(base_mean * pct, base_sd)
        # The simulation check at the formula's own answer, to confirm the closed form.
        empirical = simulate_power(base_mean * pct, base_sd, res.n_per_arm, seed=SEED)
        margin_power[f"{pct:.0%}"] = {
            "effect_mxn": round(base_mean * pct, 2),
            "effect_size": round(res.effect_size, 4),
            "n_per_arm": res.n_per_arm,
            "formula_power": res.power,
            "simulated_power": round(empirical, 3),
        }

    # (c) The question that actually matters commercially: does a markdown sell MORE?
    control_units = (frame[~frame["treated_sku"]]
                     .groupby(["item_id", "month"])["quantity"].sum())
    unit_mean, unit_sd = float(control_units.mean()), float(control_units.std())
    treated_sku_months = int(frame.loc[frame["treated_sku"]].groupby(["item_id", "month"]).ngroups)

    volume_power = {}
    for lift in (0.05, 0.10, 0.20, 0.30):
        res = required_sample_size(unit_mean * lift, unit_sd)
        volume_power[f"{lift:.0%}"] = {
            "lift_units": round(unit_mean * lift, 2),
            "effect_size": round(res.effect_size, 4),
            "n_per_arm": res.n_per_arm,
        }

    print()
    print("-" * 78)
    print("4. WHAT COULD THIS DESIGN EVER HAVE DETECTED?")
    print("-" * 78)
    print(f"  treated lines in the treated cell        : {treated_lines}")
    print(f"  treated unit margin (normal months)      : mean {base_mean:,.2f}  sd {base_sd:,.2f}")
    print()
    print("  (a) the MARGIN effect, prospectively — lines per arm required:")
    for label, row in margin_power.items():
        print(f"      detect a {label:>4} change ({row['effect_mxn']:>8,.2f} MXN): "
              f"n = {row['n_per_arm']:>6,}   "
              f"formula power {row['formula_power']:.2f} / simulated {row['simulated_power']:.2f}")
    print()
    print(f"  (b) the DEMAND response — units per SKU-month: mean {unit_mean:,.2f} sd {unit_sd:,.2f}")
    print(f"      available treated SKU-months: {treated_sku_months}")
    for label, row in volume_power.items():
        print(f"      detect a {label:>4} lift in units ({row['lift_units']:>5.2f}): "
              f"n = {row['n_per_arm']:>6,} SKU-months per arm")
    print()
    print("  ! The generator allocates units from family volumes, seasonality and velocity. PRICE")
    print("  ! NEVER ENTERS. So the markdown cannot change how much is sold, and no test on this")
    print("  ! data can answer the demand question. That is a property of the world, not a gap in")
    print("  ! the analysis — and it is the most important limitation of the whole exercise.")

    # ---- 5. The decision ---------------------------------------------------------------------
    decision = (
        f"The markdown costs MXN {abs(truth):,.2f} of margin per unit, and the effect is measured to "
        f"within {abs(did_error / truth):.0%} of the known truth. The retrospective design is "
        f"adequate for the margin question and CANNOT answer the demand question, because no "
        f"elasticity exists in this world. Decision: treat the markdown SKUs as a cost decision "
        f"(already flagged by holding-360's finding F-04), not an experiment decision — and if the "
        f"business wants to know whether clearance sells more, it needs a prospective test with "
        f"roughly {volume_power['10%']['n_per_arm']:,} SKU-months per arm to detect a 10% lift."
    )

    print()
    print("-" * 78)
    print("5. DECISION")
    print("-" * 78)
    for line in _wrap(decision, 74):
        print("  " + line)

    # ---- Write the payload -------------------------------------------------------------------
    payload = {
        "meta": {
            "source": "holding-360 published ground truth (synthetic)",
            "synthetic_data": True,
            "line_filter": "BTM customer invoices only",
            "treated_skus": list(TREATED_SKUS),
            "treated_months": list(TREATED_MONTHS),
            "seed": SEED,
        },
        "panel": {
            "lines": int(len(frame)),
            "months": int(frame["month"].nunique()),
            "period": [frame["month"].min(), frame["month"].max()],
            "control_lines": int((~frame["treated_sku"]).sum()),
            "treated_lines": int(frame["treated_sku"].sum()),
            "discounted_lines": int(len(discounted)),
            "mislabelled_lines": mislabelled,
            "cell_counts": {f"{k[0]}_{k[1]}": int(v) for k, v in est.counts.items()},
        },
        "effect": {
            "cell_means": {f"{k[0]}_{k[1]}": round(v, 2) for k, v in est.cell_means.items()},
            "naive_own_change": round(est.treated_change, 2),
            "control_change": round(est.control_change, 2),
            "did_estimate": round(est.did, 2),
            "truth_att": round(truth, 2),
            "naive_abs_error": round(abs(naive_error), 2),
            "did_abs_error": round(abs(did_error), 2),
            "welch_t": round(t_stat, 3),
            "welch_p": float(p_value),
            "bootstrap_ci": {
                "point": round(point, 2),
                "low": round(ci_low, 2),
                "high": round(ci_high, 2),
                "covers_truth": bool(ci_low <= truth <= ci_high),
            },
        },
        "parallel_trends": {
            "control_raw_sd": round(float(mix["raw_mean"].std()), 2),
            "control_fixed_weight_sd": round(float(mix["fixed_weight_mean"].std()), 2),
            "control_fixed_weight_value": round(float(mix["fixed_weight_mean"].mean()), 2),
            "mix_gap_sd": round(float(mix["mix_gap"].std()), 2),
            "placebo_did": round(placebo.did, 2),
            "placebo_ratio": round(abs(placebo.did / est.did), 4),
            "verdict": (
                "no time-varying common shock exists; the second difference removes a mix "
                "artefact worth MXN "
                f"{abs(est.control_change):,.2f} against an effect of MXN {abs(truth):,.2f}"
            ),
        },
        "power": {
            "retrospective_treated_lines": treated_lines,
            "margin": margin_power,
            "volume": volume_power,
            "volume_inputs": {
                "unit_mean": round(unit_mean, 2),
                "unit_sd": round(unit_sd, 2),
                "available_treated_sku_months": treated_sku_months,
            },
            "price_elasticity_present": False,
        },
        "decision": decision,
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "results.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print()
    print(f"wrote {out.relative_to(PROJECT_ROOT)}")

    # Also publish the headline numbers as CSV. JSON needs a parser; the sibling R lab uses base R,
    # which has read.csv and no JSON reader. A flat CSV is what makes the cross-language check in
    # `r-inference-lab/R/benchmark.R` possible without adding a dependency to either project.
    headlines = {
        "truth_att": truth,
        "naive_own_change": est.treated_change,
        "control_change": est.control_change,
        "did": est.did,
        "welch_t": t_stat,
        "welch_p": p_value,
        "control_raw_sd": float(mix["raw_mean"].std()),
        "control_fixed_weight_sd": float(mix["fixed_weight_mean"].std()),
        "placebo_did": placebo.did,
        "n_10pct_margin_effect": margin_power["10%"]["n_per_arm"],
        "n_5pct_margin_effect": margin_power["5%"]["n_per_arm"],
        "n_15pct_margin_effect": margin_power["15%"]["n_per_arm"],
        "n_10pct_volume_effect": volume_power["10%"]["n_per_arm"],
        "treated_lines_in_cell": treated_lines,
        "panel_lines": len(frame),
    }
    csv_path = REPORTS / "headline_numbers.csv"
    with csv_path.open("w") as handle:
        handle.write("metric,value\n")
        for metric, value in headlines.items():
            handle.write(f"{metric},{value}\n")
    print(f"wrote {csv_path.relative_to(PROJECT_ROOT)}")
    return payload


def _wrap(text: str, width: int) -> list[str]:
    """Wrap a sentence into lines no longer than `width`, on word boundaries."""
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


if __name__ == "__main__":
    main()
