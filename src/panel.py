"""
panel.py — build the SKU x month panel that the experiment analysis runs on.

WHY THIS FILE EXISTS
--------------------
Everything downstream (the difference-in-differences estimate, the power calculation, the notebook)
reads the same table. Building that table in one place means there is exactly one definition of
"treated", one definition of "unit margin", and no chance that two analyses disagree about which
rows are in scope.

WHAT A READER NEEDS TO KNOW FIRST
---------------------------------
* The data is **synthetic**. It comes from the sibling project `holding-360`, which generates a
  fictional three-company Mexican import group. Nothing here is client data, and no number here
  describes a real business.
* A **line** is one SKU on one invoice. A **panel cell** is one SKU in one month.
* **Treated SKU** = carries a clearance markdown (`config.MARKDOWN_PCT_BY_SKU`, four SKUs).
* **Treated month** = March, September or November (`config.MARKDOWN_MONTHS`).
* **Unit margin** = margin divided by quantity, so a 2-unit line and an 8-unit line are comparable.

THE ONE FILTER THAT MATTERS
---------------------------
`holding-360` writes two kinds of BTM document: **customer invoices** (money in, priced in MXN) and
**supplier invoices** (money out, priced in USD or CNY). Both have rows in `document_line.csv`.

An early version of this panel kept both and the result was quietly wrong: supplier lines are costs
converted at a transaction-date exchange rate, so they inject cost variation that does not exist in
the sales data, and they inflated the treated-group's apparent cost variability. The margin question
is about SALES, so the panel keeps customer invoices only. The count is asserted in the tests.
"""

from __future__ import annotations

# dataclasses: a small typed container, so the panel travels with its own metadata.
from dataclasses import dataclass
# importlib.util: load the sibling project's config by file path (see the note below on why).
import importlib.util
# pathlib: file paths as objects, so this works from any working directory.
from pathlib import Path
# sys: the module registry — the config must be registered before it is executed.
import sys

# pandas: the tabular work.
import pandas as pd

# --- Locate the sibling project ----------------------------------------------------------------
# __file__ is src/panel.py -> .parent is src/ -> .parent.parent is the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Every portfolio project sits side by side, so holding-360 is one level up.
H360_ROOT = PROJECT_ROOT.parent / "holding-360"
GROUND_TRUTH = H360_ROOT / "data" / "ground_truth"

# Import the treatment definition from the generator's own config instead of re-typing it. A
# hardcoded copy of "which SKUs carry a markdown" would silently drift from the generator, and the
# analysis would end up measuring a world that no longer exists.
#
# ! Loaded BY FILE PATH, not by `import src.config`. Both this project and holding-360 have a `src`
# ! package, so `from src import config` resolves to whichever `src` is on the path first — and here
# ! that is this project's own, which has no config.py. Importing the file directly sidesteps the
# ! collision entirely and works regardless of how the caller set up sys.path.
_CONFIG_PATH = H360_ROOT / "src" / "config.py"
_spec = importlib.util.spec_from_file_location("h360_config", _CONFIG_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(
        f"could not load the holding-360 config from {_CONFIG_PATH}. "
        "The experiment panel depends on the generator's own treatment definition."
    )
h360_config = importlib.util.module_from_spec(_spec)
# ! Register the module in sys.modules BEFORE executing it. `dataclasses` resolves a class's own
# ! module through sys.modules when it inspects annotations, so an unregistered module raises
# ! "'NoneType' object has no attribute '__dict__'" from deep inside the dataclass decorator — a
# ! confusing error that has nothing to do with the actual mistake.
sys.modules["h360_config"] = h360_config
_spec.loader.exec_module(h360_config)


#: The four SKUs that carry a clearance markdown, read from the generator's config.
TREATED_SKUS: tuple[str, ...] = tuple(h360_config.MARKDOWN_PCT_BY_SKU)
#: The calendar months in which the markdown is active: (3, 9, 11).
TREATED_MONTHS: tuple[int, ...] = tuple(h360_config.MARKDOWN_MONTHS)
#: The configured markdown depth per treated SKU, used to compute the known true effect.
MARKDOWN_PCT_BY_SKU: dict[str, float] = dict(h360_config.MARKDOWN_PCT_BY_SKU)


@dataclass
class Panel:
    """The analysis table plus the facts a reader needs to interpret it.

    Attributes
    ----------
    frame:
        One row per customer-invoice line, with the derived columns described in `build_panel`.
    truth_att:
        The true average effect of the markdown on unit margin, in MXN. It is computable exactly
        because the world is synthetic: each treated SKU's configured markdown times its list
        price, weighted by how many discounted lines that SKU actually produced.
    """

    frame: pd.DataFrame
    truth_att: float

    @property
    def treated(self) -> pd.DataFrame:
        """Lines on a treated SKU."""
        return self.frame[self.frame["treated_sku"]]

    @property
    def control(self) -> pd.DataFrame:
        """Lines on a control SKU — any SKU that never carries a markdown."""
        return self.frame[~self.frame["treated_sku"]]


def load_panel() -> Panel:
    """Read holding-360's published ground truth and return the experiment panel.

    Returns
    -------
    Panel
        The line-level frame, plus the known true treatment effect.

    Raises
    ------
    FileNotFoundError
        If the sibling project's ground-truth CSVs are not present. Failing loudly is deliberate:
        a silently empty panel would produce a confident zero.
    """
    # One row per invoice line: the unit of analysis.
    lines = pd.read_csv(GROUND_TRUTH / "document_line.csv")
    # One row per invoice: supplies the date, the company and — crucially — the document type.
    documents = pd.read_csv(GROUND_TRUTH / "document.csv")

    # Attach the header fields to each line. `validate="many_to_one"` makes pandas raise if a
    # document_id ever appears twice in the header table, which would silently multiply rows.
    panel = lines.merge(
        documents[["document_id", "company_code", "doc_type", "period_month"]],
        on="document_id",
        how="left",
        validate="many_to_one",
    )

    # bringToMX is the only company that applies markdowns, and only customer invoices are sales.
    panel = panel[(panel["company_code"] == "BTM")
                  & (panel["doc_type"] == "customer_invoice")].copy()

    # Month as "YYYY-MM" for grouping, and as a number (1-12) because the markdown rule keys on the
    # calendar month, not on a position in the series.
    period = pd.to_datetime(panel["period_month"])
    panel["month"] = period.dt.to_period("M").astype(str)
    panel["month_number"] = period.dt.month

    # The two design columns.
    panel["treated_sku"] = panel["item_id"].isin(TREATED_SKUS)
    panel["treated_period"] = panel["month_number"].isin(TREATED_MONTHS)

    # Margin per unit. `np.where` guards the zero-quantity case: dividing would give inf, which
    # propagates silently through every later mean.
    panel["unit_margin"] = (
        panel["line_margin_mxn"] / panel["quantity"].where(panel["quantity"] > 0)
    )
    # Unit cost per unit, used by the parallel-trends diagnostic to prove there is no cost shock.
    panel["unit_cost"] = (
        panel["line_cost_mxn"] / panel["quantity"].where(panel["quantity"] > 0)
    )
    # A boolean rather than a float, so "was this line discounted?" cannot be answered by 0.0001.
    panel["discounted"] = panel["discount_pct"].fillna(0) > 0

    # The generator's own published economics table supplies the LIST price per SKU.
    economics = pd.read_csv(GROUND_TRUTH / "item_economics.csv").set_index("item_id")

    return Panel(frame=panel, truth_att=_true_att(panel, economics))


def _true_att(panel: pd.DataFrame, economics: pd.DataFrame) -> float:
    """Compute the known true average effect of the markdown on unit margin.

    HOW THIS IS KNOWN RATHER THAN ESTIMATED
    ---------------------------------------
    The markdown is a mechanical rule: a treated SKU's price becomes `list_price x (1 - markdown)`
    during a clearance month. Landed cost does not move. So the true effect on one unit of SKU `s`
    is exactly `-list_price_s x markdown_s` — no inference required.

    The average effect over the treated lines the generator actually produced weights each SKU by
    how many discounted lines it has, because that is the population the estimator is averaging over.

    Parameters
    ----------
    panel:
        The line-level frame, filtered inside to treated SKU x treated period lines.
    economics:
        `item_economics.csv` indexed by `item_id`. Its `list_price_mxn` is the undiscounted price.

    Returns
    -------
    float
        The true average treatment effect on the treated, in MXN per unit. Negative: margin falls.
    """
    treated = panel[panel["treated_sku"] & panel["treated_period"]]
    if treated.empty:
        return 0.0

    # ! The list price must come from the economics table, NOT from the panel. Every treated line in
    # ! a treated period is already discounted, so a max() over those rows would return the DISCOUNTED
    # ! price and understate the cut by exactly the markdown rate. This was the first version's bug.
    list_price = treated["item_id"].map(economics["list_price_mxn"])
    markdown = treated["item_id"].map(MARKDOWN_PCT_BY_SKU)
    per_unit_cut = list_price * markdown

    # ! Weight by SKU, on a SKU index. The first version multiplied a LINE-indexed series by an
    # ! ITEM-indexed count series; pandas aligned the two indexes, produced all-NaN, and the "truth"
    # ! silently came out as 0.0 — which then divided by zero downstream. Build both sides on the
    # ! same key.
    cut_by_sku = per_unit_cut.groupby(treated["item_id"]).first()
    share_by_sku = treated["item_id"].value_counts()

    return float(-(cut_by_sku * share_by_sku).sum() / share_by_sku.sum())
