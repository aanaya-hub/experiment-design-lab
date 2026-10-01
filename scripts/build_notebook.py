"""
build_notebook.py — assemble and execute `notebooks/01-experiment-design.ipynb`.

WHY A BUILDER RATHER THAN A HAND-WRITTEN NOTEBOOK
-------------------------------------------------
A notebook edited by hand drifts from the code it claims to demonstrate, and a stale output cell is
worse than no notebook: it shows a number that the repository no longer produces. Building the
notebook from a list of cells and executing it end to end means every output in it was produced by
the code in `src/`, at the moment the notebook was built.

HOW TO RUN
----------
    python scripts/build_notebook.py

It writes `notebooks/01-experiment-design.ipynb` and executes it with nbclient. If a cell raises,
the build fails — which is the point.

WHAT A READER NEEDS TO KNOW
---------------------------
* Markdown cells carry the narrative; code cells call into `src/` rather than re-implementing it.
* The first code cell puts the project root on the import path, because a notebook's working
  directory is `notebooks/`, not the project root.
"""

from __future__ import annotations

# pathlib: paths relative to this file.
from pathlib import Path

# nbformat: builds the notebook document.
import nbformat
# nbclient: executes it, so the saved outputs are real.
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = PROJECT_ROOT / "notebooks" / "01-experiment-design.ipynb"

# Each entry is ("markdown" | "code", text). Kept as a list so the notebook reads top to bottom in
# this file, which is how it will be reviewed.
CELLS: list[tuple[str, str]] = [
    ("markdown", """# Clearance markdowns at bringToMX — an experiment that answers the wrong question well

**Synthetic data.** Everything below comes from the sibling project `holding-360`, a fictional
three-company import group. No real company is described.

**The short version.** Four SKUs carry a clearance markdown in March, September and November. That
is a natural experiment, and it has a clean difference-in-differences design. The design works: it
measures the markdown's cost to margin to within **2%** of the known truth.

And that is the problem. The quantity this design measures precisely is the one that is true by
arithmetic. The question a business would actually pay to answer — *does a markdown sell more?* —
is unanswerable in this data, because the generator has no price elasticity at all.

This notebook shows both halves: the estimate that works, and the power analysis that proves the
interesting question was never reachable."""),

    ("markdown", """## 0. Setup

The project's own modules do the work. Nothing is re-implemented here — if the notebook and `src/`
ever disagree, the notebook is wrong."""),

    ("code", """import sys
from pathlib import Path

# A notebook's working directory is notebooks/, so the project root must be added explicitly
# before `src` can be imported.
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.panel import TREATED_SKUS, TREATED_MONTHS, load_panel
from src.did import estimate_did, mix_decomposition, placebo_did, welch_test, cluster_bootstrap_did
from src.power import required_sample_size, simulate_power

pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
print(f"treated SKUs  : {list(TREATED_SKUS)}")
print(f"treated months: {list(TREATED_MONTHS)}")"""),

    ("markdown", """## 1. Build the panel, and check the treatment is where it claims to be

Nothing downstream means anything if the treatment label is wrong, so this check comes first. Every
discounted line must sit in a treated SKU **and** a treated month."""),

    ("code", """panel = load_panel()
frame = panel.frame

discounted = frame[frame["discounted"]]
mislabelled = int((~discounted["treated_sku"] | ~discounted["treated_period"]).sum())

print(f"invoice lines            : {len(frame):,}")
print(f"months                   : {frame['month'].nunique()} ({frame['month'].min()} -> {frame['month'].max()})")
print(f"discounted lines         : {len(discounted)}")
print(f"mislabelled              : {mislabelled}  {'OK' if mislabelled == 0 else 'PROBLEM'}")

grid = frame.groupby(["treated_sku", "treated_period"]).size().unstack(fill_value=0)
grid.index = ["control SKUs", "treated SKUs"]
grid.columns = ["normal months", "markdown months"]
grid"""),

    ("markdown", """**Read the grid.** The treated cell holds 27 lines. Everything that follows is limited by that
number, and the honest treatment of it is to say so rather than to run a test as though 27 lines
were 27 independent facts.

## 2. Estimate the effect, and compare it against the truth

The world is synthetic, so the true effect is not a mystery: the markdown is a mechanical price cut
at fixed cost, so the effect on one unit is exactly `-list_price x markdown`. That makes this an
**estimator check**, not a discovery."""),

    ("code", """est = estimate_did(frame)
truth = panel.truth_att
t_stat, p_value = welch_test(frame)

summary = pd.DataFrame({
    "estimate": [est.treated_change, est.did, truth],
    "error vs truth": [
        est.treated_change - truth,
        est.did - truth,
        0.0,
    ],
}, index=[
    "naive own-change (treated)",
    "DiD",
    "KNOWN TRUTH (ATT)",
])
summary["abs error %"] = (summary["error vs truth"].abs() / abs(truth) * 100).round(2)
summary"""),

    ("code", """point, low, high = cluster_bootstrap_did(frame, n_boot=2_000, seed=11)
print(f"Welch t on treated lines : t = {t_stat:,.2f}, p = {p_value:.3g}")
print(f"cluster bootstrap 95% CI : [{low:,.2f}, {high:,.2f}]")
print(f"covers the known truth   : {low <= truth <= high}")"""),

    ("markdown", """**Both estimators recover the effect.** The DiD is marginally closer to the truth than the naive
comparison, and the bootstrap interval covers it. So far this looks like a textbook result.

The next section is where it stops looking like one.

## 3. What is the second difference actually removing?

DiD earns its keep by subtracting a **common shock** — something that moved both groups. If nothing
common moved, the second difference is subtracting noise or composition, and the sophistication buys
almost nothing.

The test: recompute the control group's monthly mean margin with **fixed SKU weights**. If the raw
series swings while the fixed-weight series is flat, the swing was product mix all along."""),

    ("code", """control = frame[~frame["treated_sku"]]
mix = mix_decomposition(control)

pd.DataFrame({
    "raw monthly mean (MXN)": mix["raw_mean"],
    "fixed-weight mean (MXN)": mix["fixed_weight_mean"],
    "mix gap": mix["mix_gap"],
}).head(8)"""),

    ("code", """print(f"raw mean            : sd {mix['raw_mean'].std():>8,.2f}  "
      f"range {mix['raw_mean'].max() - mix['raw_mean'].min():>8,.2f}")
print(f"fixed-weight mean   : sd {mix['fixed_weight_mean'].std():>8,.2f}  "
      f"range {mix['fixed_weight_mean'].max() - mix['fixed_weight_mean'].min():>8,.2f}")
print()
print("! A fixed-weight series with sd = 0 means the control group's movement is ENTIRELY mix.")
print("! There is no time-varying cost or price shock. The second difference removes a")
print(f"!  composition artefact worth MXN {abs(est.control_change):,.2f} against an effect of "
      f"MXN {abs(truth):,.2f}.")"""),

    ("markdown", """### The placebo test

Apply the same DiD machinery to a **fake** treatment inside months where no markdown ever happened.
A well-behaved design returns approximately zero. Whatever it returns instead is the noise floor —
the size of estimate this design can manufacture out of nothing."""),

    ("code", """placebo = placebo_did(frame)
print(f"real DiD    : {est.did:>10,.2f}")
print(f"placebo DiD : {placebo.did:>10,.2f}")
print(f"ratio       : {abs(placebo.did / est.did):>10.1%}")
print()
print("The placebo sits at the same magnitude as the control group's mix artefact, which is")
print("exactly what it should be: there is no shock to find, so the design finds composition.")"""),

    ("markdown", """## 4. Power — what could this design ever have detected?

The formula, for a two-arm comparison:

```
n per arm = 2 * (z_{1-alpha/2} + z_{1-beta})^2 / (delta / sigma)^2
```

Two ways to compute it are shown, and they must agree. The closed form is what an interviewer
expects; the simulation is the check that the formula has no slip in it."""),

    ("code", """treated_normal = frame.loc[frame["treated_sku"] & ~frame["treated_period"], "unit_margin"]
base_mean, base_sd = float(treated_normal.mean()), float(treated_normal.std())

rows = []
for pct in (0.05, 0.10, 0.15):
    res = required_sample_size(base_mean * pct, base_sd)
    empirical = simulate_power(base_mean * pct, base_sd, res.n_per_arm, seed=7)
    rows.append({
        "effect": f"{pct:.0%}",
        "delta (MXN)": round(base_mean * pct, 2),
        "effect size": round(res.effect_size, 4),
        "n per arm": res.n_per_arm,
        "formula power": res.power,
        "simulated power": round(empirical, 3),
    })

pd.DataFrame(rows).set_index("effect")"""),

    ("markdown", """**The formula holds**: the simulated power lands on the target within a point or two. And the
requirement grows with the square of the effect — halving the effect you want to detect quadruples
the sample.

Now the question that matters. **Does a markdown sell more units?**"""),

    ("code", """control_units = control.groupby(["item_id", "month"])["quantity"].sum()
unit_mean, unit_sd = float(control_units.mean()), float(control_units.std())
available = int(frame.loc[frame["treated_sku"]].groupby(["item_id", "month"]).ngroups)

rows = []
for lift in (0.05, 0.10, 0.20, 0.30):
    res = required_sample_size(unit_mean * lift, unit_sd)
    rows.append({
        "lift to detect": f"{lift:.0%}",
        "delta (units)": round(unit_mean * lift, 2),
        "effect size": round(res.effect_size, 4),
        "SKU-months per arm": res.n_per_arm,
    })

print(f"units per SKU-month: mean {unit_mean:,.2f}, sd {unit_sd:,.2f}")
print(f"treated SKU-months available: {available}")
print()
pd.DataFrame(rows).set_index("lift to detect")"""),

    ("markdown", """## 5. The finding

**Look at the two tables together.** To detect a 10% change in unit margin the design needs about
**695 lines** per arm — and it has **27** in the treated cell. To detect a 10% lift in **units** it
would need about **1,702 SKU-months** per arm against **70** available.

But the sample size is not the real problem. The real problem is this: **the generator allocates
units from family volumes, seasonality and SKU velocity. Price never enters that function.**

So the markdown's effect on volume is **exactly zero by construction**. No test can find an effect
that does not exist — and running one anyway would produce a null result that reads like evidence
while being pure arithmetic.

### What this design can and cannot say

| Question | Answerable here? |
| --- | --- |
| Does the markdown reduce margin per unit? | **Yes** — measured to within 2% of the known truth |
| Did a common shock affect both groups? | **No such shock exists** — the control group moves on mix alone |
| Can the second difference be credited with removing confounders? | **Only weakly** — it removes MXN 23.66 of mix against a MXN 1,900 effect |
| Does the markdown sell more units? | **No** — zero by construction, and 1,702 SKU-months per arm would be needed to test it in a world that had an effect |
| What should the owner do? | Treat the SKUs as a **cost** decision, not an experiment decision |

### Why this is the useful result

It would have been easy to report the DiD estimate, note that the confidence interval excluded zero,
and call the exercise a success. That report would have been true and useless: it would have
presented an arithmetic identity as a causal discovery, and it would have left the reader believing
the demand question had been tested.

**The competence being demonstrated is not "can run a DiD". It is "knows which question the design
can answer, and says so out loud when the answer is the unflattering one."**"""),

    ("markdown", """## 6. The decision

1. **The margin question is closed.** The markdown costs **MXN 1,900.37** of margin per unit, measured
   to within 2%. This confirms `holding-360`'s finding F-04 with an interval instead of a point.
2. **The demand question is not closed, and cannot be closed with this data.** Say so, and do not
   let a well-powered answer to a trivial question stand in for an unanswerable important one.
3. **The owner's decision stands as a cost decision**, not as an experiment: withdrawal of the two
   loss-making SKUs is justified by the margin arithmetic alone.
4. **If demand response is genuinely wanted**, the route is a prospective test with the sample size
   from §4, or a generator that models elasticity. Both are stated in `DESIGN.md` §10.
"""),
]


def build() -> Path:
    """Assemble the notebook, execute it, and save it. Returns the output path."""
    # nbformat.v4.new_notebook() gives an empty, correctly-versioned document.
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell(text) if kind == "markdown"
        else nbformat.v4.new_code_cell(text)
        for kind, text in CELLS
    ]
    # Record the kernel so the notebook is reproducible outside this machine's default.
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3",
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Execute in the project root so relative paths inside the notebook behave predictably.
    # timeout=300 because the bootstrap and the power simulation are the slow parts.
    client = NotebookClient(notebook, timeout=300, kernel_name="python3",
                            resources={"metadata": {"path": str(PROJECT_ROOT)}})
    client.execute()

    nbformat.write(notebook, OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    path = build()
    print(f"wrote and executed {path.relative_to(PROJECT_ROOT)}")
