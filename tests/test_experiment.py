"""
test_experiment.py — the assertions that hold the analysis honest.

WHY THESE TESTS EXIST
---------------------
Two kinds of test are here, and the distinction matters:

1. **Integrity tests** — the treatment is where the design says it is, the panel contains only sales
   lines, and the known truth is the value the arithmetic says it is. If one of these fails, every
   result downstream is meaningless, so they run first.
2. **Behaviour tests** — the estimator recovers a known answer on a hand-built example, the control
   group's movement really is pure mix, and the sample-size formula agrees with a simulation. These
   are the ones that catch a silent regression in the logic.

The most valuable test in this file is `test_control_group_movement_is_entirely_mix`. It asserts a
finding, not an implementation detail: if someone later changes the generator to add a real cost
shock, that test fails and tells them the parallel-trends conclusion no longer holds.
"""

# pandas: build the small hand-made frame used by the estimator tests.
import pandas as pd
# pytest: for the approx helper and the raises check.
import pytest

from src.panel import TREATED_MONTHS, TREATED_SKUS, load_panel
from src.power import required_sample_size, simulate_power
from src.did import cluster_bootstrap_did, estimate_did, mix_decomposition, placebo_did


@pytest.fixture(scope="module")
def panel():
    """The panel, built once for the whole module — it reads CSVs and is not cheap."""
    return load_panel()


# --- 1. Integrity -----------------------------------------------------------------------------

def test_treatment_is_exactly_where_the_design_says_it_is(panel):
    """Every discounted line must sit in a treated SKU x treated month cell. No exceptions."""
    discounted = panel.frame[panel.frame["discounted"]]
    assert len(discounted) > 0, "no discounted lines at all — the design has nothing to measure"
    assert (discounted["treated_sku"]).all(), "a control SKU carried a markdown"
    assert (discounted["treated_period"]).all(), "a markdown landed outside a clearance month"


def test_panel_contains_only_sales_lines(panel):
    """Supplier invoices are costs in USD/CNY and must not be in a margin panel.

    The first version included them and inflated the treated group's apparent cost variability.
    """
    assert "supplier_invoice" not in set(panel.frame["doc_type"].unique())
    assert set(panel.frame["doc_type"].unique()) == {"customer_invoice"}


def test_known_truth_matches_hand_arithmetic(panel):
    """Recompute the true ATT independently and require the module to agree.

    The arithmetic: each treated SKU's list price times its configured markdown, weighted by how
    many discounted lines that SKU produced.
    """
    treated = panel.frame[panel.frame["treated_sku"] & panel.frame["treated_period"]]
    list_prices = {"BTM-1111": 11_350.0, "BTM-1112": 6_050.0,
                   "BTM-2204": 8_850.0, "BTM-2205": 10_700.0}
    pcts = {"BTM-1111": 0.15, "BTM-1112": 0.15, "BTM-2204": 0.20, "BTM-2205": 0.25}

    expected = 0.0
    total = 0
    for sku, pct in pcts.items():
        n = int((treated["item_id"] == sku).sum())
        expected -= list_prices[sku] * pct * n
        total += n

    assert total == 27, f"expected 27 discounted lines, found {total}"
    assert panel.truth_att == pytest.approx(expected / total, rel=1e-9)
    assert panel.truth_att < 0, "the markdown cuts margin; the truth must be negative"


def test_every_treated_sku_has_a_configured_markdown(panel):
    """The four treated SKUs are the four the generator configures — no more, no fewer."""
    assert set(TREATED_SKUS) == {"BTM-1111", "BTM-1112", "BTM-2204", "BTM-2205"}
    assert set(TREATED_MONTHS) == {3, 9, 11}


# --- 2. The estimator on a hand-built example -------------------------------------------------

def test_did_recovers_a_known_answer_on_a_hand_built_frame():
    """Four cells, four means, an answer computable by hand — the test of the arithmetic itself.

    Treated:  10 then 6   -> own change -4
    Control:  10 then 9   -> own change -1
    DiD = -4 - (-1) = -3
    """
    frame = pd.DataFrame({
        "treated_sku":    [True] * 2 + [False] * 2,
        "treated_period": [False, True, False, True],
        "unit_margin":    [10.0, 6.0, 10.0, 9.0],
    })
    result = estimate_did(frame)
    assert result.treated_change == pytest.approx(-4.0)
    assert result.control_change == pytest.approx(-1.0)
    assert result.did == pytest.approx(-3.0)


def test_did_raises_when_a_cell_is_empty():
    """A missing cell makes the comparison undefined; returning a number would be worse than failing."""
    frame = pd.DataFrame({
        "treated_sku":    [True, False],
        "treated_period": [False, False],
        "unit_margin":    [10.0, 9.0],
    })
    with pytest.raises(ValueError, match="2x2 grid is incomplete"):
        estimate_did(frame)


# --- 3. The findings the analysis reports -----------------------------------------------------

def test_control_group_movement_is_entirely_mix(panel):
    """! This asserts the project's central methodological finding.

    Control-group unit margin is constant per SKU (fixed price, fixed landed cost), so a
    fixed-weight series must be exactly flat. If it ever moves, a real cost or price shock has
    appeared in the generator and the parallel-trends verdict has to be re-derived.
    """
    mix = mix_decomposition(panel.frame[~panel.frame["treated_sku"]])
    assert mix["raw_mean"].std() > 50, "the raw series should visibly move (it is the mix effect)"
    assert mix["fixed_weight_mean"].std() == pytest.approx(0.0, abs=1e-9), (
        "the fixed-weight control series moved, so a real common shock now exists — "
        "the parallel-trends conclusion no longer holds and must be re-derived"
    )


def test_placebo_effect_is_small_relative_to_the_real_effect(panel):
    """The design must not manufacture a large number out of nothing."""
    real = estimate_did(panel.frame).did
    fake = placebo_did(panel.frame).did
    assert abs(fake / real) < 0.10, (
        f"the placebo estimate ({fake:,.2f}) is more than 10% of the real one ({real:,.2f}); "
        "the design is producing composition artefacts at a size that matters"
    )


def test_did_is_closer_to_the_truth_than_the_naive_comparison(panel):
    """The justification for the second difference, stated as an assertion.

    If DiD were ever WORSE than the naive change, the extra machinery would be unjustified and the
    README would have to say so.
    """
    est = estimate_did(panel.frame)
    truth = panel.truth_att
    assert abs(est.did - truth) <= abs(est.treated_change - truth)


def test_bootstrap_interval_covers_the_truth(panel):
    """A 95% interval that misses a known truth would mean the inference is broken."""
    _, low, high = cluster_bootstrap_did(panel.frame, n_boot=400, seed=11)
    assert low <= panel.truth_att <= high, (
        f"bootstrap CI [{low:,.2f}, {high:,.2f}] excludes the known truth {panel.truth_att:,.2f}"
    )
    assert low < high, "the interval is inverted or degenerate"


# --- 4. The power calculation -----------------------------------------------------------------

def test_closed_form_matches_simulation():
    """The formula is only trustworthy if a simulation agrees with it.

    A mis-signed term or a one-sided/two-sided slip changes the answer by enough to show up here.
    """
    for effect, sd in ((100.0, 1000.0), (250.0, 1000.0)):
        result = required_sample_size(effect, sd)
        empirical = simulate_power(effect, sd, result.n_per_arm, n_trials=1_500, seed=3)
        assert empirical == pytest.approx(result.power, abs=0.05), (
            f"formula says power {result.power:.2f}, simulation measured {empirical:.2f}"
        )


def test_halving_the_effect_quadruples_the_sample():
    """The single most important property of the formula, and the one that drives the conclusions."""
    big = required_sample_size(200.0, 1000.0).n_per_arm
    small = required_sample_size(100.0, 1000.0).n_per_arm
    assert small == pytest.approx(4 * big, rel=0.02)


def test_sample_size_rejects_undefined_inputs():
    """A zero variance or a zero effect would divide by zero; both must raise, not return infinity."""
    with pytest.raises(ValueError):
        required_sample_size(0.0, 1000.0)
    with pytest.raises(ValueError):
        required_sample_size(100.0, 0.0)


def test_detecting_the_real_effect_needs_far_fewer_lines_than_the_panel_has(panel):
    """The markdown's effect is large and mechanical, so the retrospective study is well powered.

    This is the flip side of the project's finding: the thing the data CAN measure, it measures
    easily — which is precisely why measuring it is not the interesting part.
    """
    needed = required_sample_size(panel.truth_att, 950.0).n_per_arm
    available = int(estimate_did(panel.frame).counts[(True, True)])
    assert available >= needed, "the mechanical effect should be detectable from the treated lines"
    assert needed < 10, f"a {abs(panel.truth_att):,.0f} MXN effect against sd 950 needs very few lines"
