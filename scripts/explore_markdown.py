"""
explore_markdown.py — measure the markdown "experiment" that already exists inside holding-360.

WHY THIS FILE EXISTS
--------------------
The portfolio plan proposed a difference-in-differences (DiD) design on the markdown months of
`holding-360`. Before writing a single line of the analysis, this script MEASURES the panel that
would be used: how many treated units exist, how many control units, which months are treated,
how spread out the outcome is, and whether the control group actually moves over time.

That last question is the one that decides whether DiD is the right tool at all. If the control
group's margin is flat, differencing it away buys nothing and a simple pre/post comparison would
do. If it moves — and here it moves because landed cost is converted at the exchange rate of the
transaction date — then DiD is removing a real common shock, which is exactly what it is for.

The rule this campaign runs on: measure the instrument before trusting the result.

HOW TO RUN
----------
    python scripts/explore_markdown.py

It reads only the published ground-truth CSVs from the sibling `holding-360` project. It writes
nothing and changes nothing — it is a read-only probe.

WHAT A READER NEEDS TO KNOW BEFORE THE CODE
-------------------------------------------
* A "unit" here is one invoice line: one SKU sold on one document on one day.
* `line_margin_mxn` is money left after landed cost. `unit_margin` divides it by the quantity, so
  a 3-unit line and an 8-unit line become comparable.
* A "treated" SKU is one in `config.MARKDOWN_PCT_BY_SKU` — four SKUs that carry a clearance
  discount. A "control" SKU is any other SKU, which never carries one.
* A "treated month" is March, September or November — the months in `config.MARKDOWN_MONTHS`.
"""

# --- Imports -----------------------------------------------------------------------------------
# pathlib: file paths as objects, so this works from any working directory.
from pathlib import Path
# sys: used to extend the import path so the sibling project's config can be read.
import sys
# pandas: all of the tabular work.
import pandas as pd
# numpy: the summary statistics that pandas does not compute directly.
import numpy as np

# --- Locate the sibling project and its data ---------------------------------------------------
# __file__ is this script; .parent is scripts/; .parent.parent is the experiment-design-lab root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# The portfolio directory holds every project side by side, so the sibling is one level up.
PORTFOLIO_ROOT = PROJECT_ROOT.parent
# holding-360 is where the synthetic world and its published ground truth live.
H360 = PORTFOLIO_ROOT / "holding-360"
GROUND_TRUTH = H360 / "data" / "ground_truth"

# Read the treatment definition from the generator's own config rather than re-typing it.
# This matters: a hardcoded copy of "which SKUs are treated" would silently drift from the
# generator and the analysis would be measuring a world that no longer exists.
sys.path.insert(0, str(H360))
from src import config as h360_config  # noqa: E402  (import must follow the sys.path edit)

# --- Load the two tables the panel is built from ----------------------------------------------
# document_line.csv: one row per invoice line — the unit of analysis.
lines = pd.read_csv(GROUND_TRUTH / "document_line.csv")
# document.csv: one row per invoice — needed for the date and the company, which the line lacks.
docs = pd.read_csv(GROUND_TRUTH / "document.csv")

# --- Join the line to its document header ------------------------------------------------------
# left join: keep every line, attach the header fields it needs. A line whose document is missing
# would mean a broken referential key, which is worth surfacing rather than dropping silently.
panel = lines.merge(
    docs[["document_id", "company_code", "doc_date", "period_month"]],
    on="document_id",
    how="left",
    validate="many_to_one",
)

# bringToMX is the only company with markdowns, so the panel is restricted to its customer invoices.
panel = panel[panel["company_code"] == "BTM"].copy()

# --- Derive the analysis columns ---------------------------------------------------------------
# period_month arrives as "2025-01-01"; keep the date for the month, strip it for grouping.
panel["month"] = pd.to_datetime(panel["period_month"]).dt.to_period("M").astype(str)
# month_number (1-12) is what the markdown rule keys on, not the month's position in the series.
panel["month_number"] = pd.to_datetime(panel["period_month"]).dt.month

# Treated SKU = it carries a markdown percentage in the generator's config.
TRATED_SKUS = set(h360_config.MARKDOWN_PCT_BY_SKU)
panel["treated_sku"] = panel["item_id"].isin(TRATED_SKUS)

# Treated period = the calendar month is one of the clearance months.
TRATED_MONTHS = tuple(h360_config.MARKDOWN_MONTHS)  # (3, 9, 11)
panel["treated_period"] = panel["month_number"].isin(TRATED_MONTHS)

# Margin per unit: a 3-unit line and an 8-unit line must be comparable.
# Guard against a zero quantity, which would make the division undefined rather than zero.
panel["unit_margin"] = np.where(
    panel["quantity"] > 0, panel["line_margin_mxn"] / panel["quantity"], np.nan
)
# Margin as a share of revenue: scale-free, so a MXN 30,000 drone and a MXN 30 cable compare.
panel["margin_pct"] = np.where(
    panel["line_total"] > 0, panel["line_margin_mxn"] / panel["line_total"], np.nan
)

# --- Report 1: the shape of the panel ----------------------------------------------------------
print("=" * 78)
print("1. PANEL SHAPE")
print("=" * 78)
print(f"BTM invoice lines            : {len(panel):,}")
print(f"Distinct months              : {panel['month'].nunique()}  "
      f"({panel['month'].min()} -> {panel['month'].max()})")
print(f"Distinct SKUs sold           : {panel['item_id'].nunique()}")
print(f"Treated SKUs in the data     : {sorted(panel.loc[panel['treated_sku'], 'item_id'].unique())}")
print(f"Control SKUs in the data     : {panel.loc[~panel['treated_sku'], 'item_id'].nunique()}")
print(f"Treated months               : {sorted(TRATED_MONTHS)}")
print()
print("Lines per cell of the 2x2 grid:")
grid = (panel.groupby(["treated_sku", "treated_period"]).size().unstack(fill_value=0))
print(grid.to_string())

# --- Report 2: did the markdown actually apply? ------------------------------------------------
# This is the cheap integrity check that stops the whole analysis being built on a mislabelled
# column. The discount column should be non-zero ONLY in treated rows.
print()
print("=" * 78)
print("2. INTEGRITY — is the treatment where the design says it is?")
print("=" * 78)
disc_check = (panel.assign(has_discount=panel["discount_pct"].fillna(0) > 0)
              .groupby(["treated_sku", "treated_period"])["has_discount"]
              .agg(["sum", "count"]))
print(disc_check.to_string())
print()
observed_rates = (panel[panel["discount_pct"].fillna(0) > 0]
                  .groupby("item_id")["discount_pct"].agg(["min", "max", "count"]))
print("Observed discount rates on discounted lines:")
print(observed_rates.to_string())
print()
print("Configured markdown rates:")
for sku, pct in h360_config.MARKDOWN_PCT_BY_SKU.items():
    print(f"  {sku}: {pct:.0%}")

# --- Report 3: the outcome, by cell of the 2x2 -------------------------------------------------
# The naive comparison: how do treated and control margins differ in treated and untreated months?
print()
print("=" * 78)
print("3. THE OUTCOME — unit margin by cell (the raw DiD ingredients)")
print("=" * 78)
cells = panel.groupby(["treated_sku", "treated_period"])["unit_margin"].agg(
    ["count", "mean", "std", "median"]
)
print(cells.to_string(float_format=lambda v: f"{v:,.2f}"))
print()
print("Cell means, as the four numbers a 2x2 DiD is built from:")
means = panel.groupby(["treated_sku", "treated_period"])["unit_margin"].mean()
try:
    t_treat = means.loc[(True, True)]
    t_ctrl = means.loc[(True, False)]
    c_treat = means.loc[(False, True)]
    c_ctrl = means.loc[(False, False)]
    print(f"  treated SKU, markdown month : {t_treat:>12,.2f}")
    print(f"  treated SKU, normal month   : {t_ctrl:>12,.2f}   -> own change {t_treat - t_ctrl:>12,.2f}")
    print(f"  control SKU, markdown month : {c_treat:>12,.2f}")
    print(f"  control SKU, normal month   : {c_ctrl:>12,.2f}   -> own change {c_treat - c_ctrl:>12,.2f}")
    print(f"  DiD estimate                : {(t_treat - t_ctrl) - (c_treat - c_ctrl):>12,.2f}")
except KeyError as exc:
    print(f"  ! a cell of the 2x2 grid is empty: {exc}")

# --- Report 4: does the control group move over time? ------------------------------------------
# THIS is what decides whether DiD earns its keep. If control margin is flat, common shocks are
# absent and DiD is decoration. If it moves, DiD is removing something real.
print()
print("=" * 78)
print("4. WHY DiD AND NOT A PRE/POST TEST — does the control group move?")
print("=" * 78)
ctrl_monthly = (panel[~panel["treated_sku"]]
                .groupby("month")["unit_margin"].agg(["mean", "count"]))
print("Control-group mean unit margin by month:")
print(ctrl_monthly.to_string(float_format=lambda v: f"{v:,.2f}"))
print()
ctrl_sd = ctrl_monthly["mean"].std()
print(f"Control-group monthly mean: sd = {ctrl_sd:,.2f} over {len(ctrl_monthly)} months")
print(f"  min {ctrl_monthly['mean'].min():,.2f}   max {ctrl_monthly['mean'].max():,.2f}")

# --- Report 5: the known truth, read from the generator's own economics ------------------------
# Because this is a synthetic world, the "true" effect of the markdown on margin is not a mystery:
# it is the configured percentage times the list price. Stating it up front is what makes the
# analysis an estimator check rather than a discovery.
print()
print("=" * 78)
print("5. THE KNOWN TRUTH — what the markdown MUST do to margin, arithmetically")
print("=" * 78)
econ = pd.read_csv(GROUND_TRUTH / "item_economics.csv")
econ = econ[econ["item_id"].isin(TRATED_SKUS)]
for _, row in econ.iterrows():
    pct = h360_config.MARKDOWN_PCT_BY_SKU[row["item_id"]]
    lost = row["list_price_mxn"] * pct
    print(f"  {row['item_id']}: list {row['list_price_mxn']:>10,.2f}  markdown {pct:>5.0%}  "
          f"-> price cut {lost:>10,.2f}  (landed {row['landed_mxn']:>10,.2f}, "
          f"normal margin {row['margin_mxn']:>9,.2f})")
print()
print("! NOTE: the generator's unit volumes do NOT depend on price (see _month_units_by_sku).")
print("! So the markdown cannot change HOW MUCH is sold, only what each unit earns.")
print("! Any 'sales went up' reading from this data would be an artefact, not a finding.")

# --- Report 6: power inputs ---------------------------------------------------------------------
# The numbers a sample-size calculation needs, taken from the data rather than assumed.
print()
print("=" * 78)
print("6. POWER ANALYSIS INPUTS")
print("=" * 78)
within = panel.groupby(["item_id", "month"])["unit_margin"].agg(["mean", "std", "count"])
print(f"SKU-month cells              : {len(within):,}")
print(f"Lines per SKU-month (median) : {within['count'].median():.0f}")
treated_within = panel[panel["treated_sku"]].groupby(["item_id", "month"])["unit_margin"].mean()
control_within = panel[~panel["treated_sku"]].groupby(["item_id", "month"])["unit_margin"].mean()
print(f"Within-cell sd, treated SKUs : {panel[panel['treated_sku']]['unit_margin'].std():,.2f}")
print(f"Within-cell sd, control SKUs : {panel[~panel['treated_sku']]['unit_margin'].std():,.2f}")
print(f"Between SKU-month sd (all)   : {within['mean'].std():,.2f}")
print()
print(f"Treated SKU-months available : {len(treated_within)}")
print(f"Control SKU-months available : {len(control_within)}")
# Means are much less noisy than individual lines — that difference is the whole point of power.
print(f"sd of treated SKU-month MEANS   : {treated_within.std():,.2f}")
print(f"sd of control SKU-month MEANS   : {control_within.std():,.2f}")

# --- Report 7: the reference effect size -------------------------------------------------------
# Expressed against the treated SKUs' own normal margin, so "detect a 5% change" is meaningful.
print()
print("=" * 78)
print("7. REFERENCE EFFECT SIZE FOR THE POWER CALCULATION")
print("=" * 78)
treated_normal = panel[panel["treated_sku"] & ~panel["treated_period"]]["unit_margin"]
print(f"Treated SKUs, normal months — mean unit margin : {treated_normal.mean():,.2f}")
print(f"                              sd               : {treated_normal.std():,.2f}")
print(f"                              n lines          : {len(treated_normal):,}")
for pct_effect in (0.05, 0.10, 0.15, 0.20, 0.25):
    print(f"  a {pct_effect:>4.0%} change in unit margin = {treated_normal.mean() * pct_effect:>10,.2f} MXN")
print()
print("Done. Nothing was written; this was a read-only probe.")
