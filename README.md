# Experiment design lab: clearance markdowns, a natural experiment

A clearance-pricing experiment, designed before it was analysed. The design, the sample size and the analysis were written down and committed before any result was looked at.

**What the project adds:** it is only possible to know whether the estimate is correct because the world is generated, and the true effect is published alongside the estimate.

**Stack:** Python · pandas · NumPy · SciPy · Jupyter · difference in differences · cluster bootstrap · Welch's t test

Synthetic data from the published ground truth of `holding-360`. No API keys, no paid services, runs offline.

---

## The design

Four SKUs received a markdown across three markdown months. Everything else stayed at price.

**Sample size was computed from the variance measured in the data, not from a textbook default.** The power calculation verified in closed form was checked against a simulation, which returned 0.80 against 0.81.

**Parallel trends was tested rather than assumed.** The mix-neutral control series has a standard deviation of 0.00 against a raw standard deviation of 72.51, which means the control group moves on product mix alone. There is no common shock for the method to remove, and the placebo estimate is 1.4 percent of the real effect.

---

## The result, against a known truth

| Method | Estimate | Distance from truth |
| --- | --- | --- |
| Naive own change | MXN 1,847.05 | 2.8 percent |
| **Difference in differences** | **MXN 1,870.71** | **1.6 percent** |
| **Known true effect** | **MXN 1,900.37** | |

A cluster bootstrap over SKUs gives a 95 percent interval of MXN 2,468.92 to MXN 1,296.04, which covers the truth. Welch's t test returns 6.90 at p of 6.0e-08.

The treatment was also verified rather than trusted: 27 discounted lines, none mislabelled, exactly where the design said they would be.

---

## What it cannot answer

**The generated world has no price elasticity.** So the design estimates the arithmetic of a markdown precisely and cannot say what would have happened to demand. That is the question a business would pay for, and the project says so instead of implying otherwise.

The power analysis is what makes this concrete. A 10 percent margin change needs 695 invoice lines per arm, which the data supports. A 10 percent volume lift needs 1,702 SKU months per arm against 70 available. The volume question was not answerable before a single estimate was computed.

---

## What is in here

`DESIGN.md`, the pre-registered design, committed before the analysis. Four source modules under `src/`. A 19 cell notebook, nine of them code, executed end to end. 14 tests. `reports/results.json` and `reports/headline_numbers.csv`.

```bash
make check    # the analysis, the tests and the notebook rebuild, as one gate
