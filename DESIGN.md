# Experiment design — clearance markdowns at bringToMX

> **Status: PRE-REGISTERED.** The design below was fixed before the analysis was run. The results
> live in `reports/results.json` and `README.md`, not here. Written 2026-09-30.
>
> **The data is synthetic.** Everything here comes from the sibling project `holding-360`, a
> fictional three-company import group in Jalisco. No real company, customer, invoice or exchange
> rate is involved, and nothing on this page may be described as client work.

## 1. Why this document exists

The screening question that ended a real application in September 2026 was:

> *"Do you have hands-on experience designing and analyzing A/B tests end to end, including
> power/sample-size calculations and significance decisions?"*

That question has five parts, and this document answers them in order: frame the hypothesis, choose
the metric, compute the sample, choose the test, and state the decision. It also answers the part
the question does not ask but a good interviewer will: **what this design cannot identify.**

## 2. The business decision this supports

`holding-360`'s finding F-04 says two SKUs sell below landed cost when their clearance markdown is
active. The owner therefore faces a real choice:

| Option | Reasoning |
| --- | --- |
| **A. Keep selling at the markdown** | Cash now, but each unit destroys margin |
| **B. Withdraw the SKUs** | Stops the loss, but locks up capital already committed |
| **C. Deepen the markdown** | Clears stock faster — *if* demand responds to price |

Option C is the one that requires evidence. It is also the one this data cannot test, for a reason
established in §7.

## 3. The hypothesis

Stated so that it can be wrong:

> **H1.** The clearance markdown reduces per-unit contribution margin on the treated SKUs.
>
> **H0.** It does not: any observed difference is explained by other factors.

H1 is deliberately unambitious. It is a **mechanical** claim — a price cut at fixed cost must reduce
per-unit margin — and it is testable here because the price change is observed directly on the
invoice line. The commercially interesting hypothesis is different and is stated in §7:

> **H2.** The markdown increases the number of units sold.

## 4. Design

| Element | Choice | Why |
| --- | --- | --- |
| **Unit** | One invoice line | It is where both price and cost are recorded |
| **Population** | bringToMX customer invoices, 2025-01 → 2026-06 | The only company applying markdowns |
| **Treated group** | The 4 SKUs in `config.MARKDOWN_PCT_BY_SKU` | Clearance stock: `BTM-2204`, `BTM-2205`, `BTM-1111`, `BTM-1112` |
| **Control group** | The 40 SKUs that never carry a markdown | Untreated, sold through the same channels in the same months |
| **Treated periods** | March, September, November (`config.MARKDOWN_MONTHS`) | The months the generator activates clearance |
| **Outcome** | `unit_margin` = invoice-line margin ÷ quantity | Comparable across a 2-unit line and an 8-unit line |
| **Estimator** | Difference-in-differences (DiD) | Removes anything affecting both groups over the same window |
| **Inference** | Cluster bootstrap over SKUs, 95% | Only 4 treated SKUs; lines of one SKU are not independent |
| **Decision rule** | Report the effect, and state whether the design can support acting on it | The number alone is not a decision |
| **α** | 0.05, two-sided | Conventional |
| **Power target** | 0.80 | Conventional |

## 5. Sample size — the calculation, before any result

The standard two-arm formula, written out because the arithmetic is the deliverable:

```
                 2 * (z_{1-α/2} + z_{1-β})²
n per arm  =  ──────────────────────────────
                     (Δ / σ)²
```

With α = 0.05 → `z` = 1.96, and power = 0.80 → `z` = 0.8416, the numerator is `2 * 7.849 = 15.70`.
So `n per arm = 15.70 / (Δ/σ)²`.

**Inputs must be measured, not assumed.** From the treated SKUs' own non-markdown lines:

| Input | Value | Source |
| --- | --- | --- |
| Baseline mean unit margin | **MXN 1,428.64** | Treated lines, non-markdown months |
| Standard deviation (σ) | **MXN 949.96** | Same lines |
| Units per SKU-month | mean **13.68**, sd **14.24** | Control SKU-months |

The sample required to detect a change in **unit margin**:

| Effect to detect | Δ (MXN) | Δ/σ | n per arm |
| --- | --- | --- | --- |
| 5% | 71.43 | 0.075 | **2,777 lines** |
| 10% | 142.86 | 0.150 | **695 lines** |
| 15% | 214.30 | 0.226 | **309 lines** |

The `(Δ/σ)²` term is the reason these numbers are large and why halving the target effect
**quadruples** the requirement. This is asserted as a test (`test_halving_the_effect_quadruples_the_sample`).

## 6. The test, and when not to use it

**The test:** Welch's t-test comparing treated lines inside markdown months against treated lines
outside them, plus the DiD estimate for the panel. Welch is chosen because it does not assume the
two groups share a variance, and nothing about this setting makes equal variances plausible.

**When not to test at all.** No test is run on the volume question (§7). Running a significance test
on a quantity that is zero by construction would produce a null result that reads like evidence and
is actually arithmetic. Declining to test is the correct answer here, and it is the harder one.

## 7. What this design cannot identify — the most important section

**The generator allocates units from family base volumes, monthly seasonality and SKU velocity.
Price never enters that function.** Measured in `scripts/explore_markdown.py` and asserted in the
tests: within a SKU, unit cost and unit price are constant, so control-group unit margin is constant
up to a rounding artefact.

Three consequences, in order of importance:

1. **There is no price elasticity in this world.** H2 — *does the markdown sell more?* — has an
   answer of exactly zero by construction. Any apparent volume response would be an artefact.
2. **There is no time-varying common shock for DiD to remove.** The control group's month-to-month
   movement is **100% product mix**: re-weighting the control group to constant SKU shares makes its
   monthly mean exactly flat. DiD's second difference is therefore correcting a composition
   artefact, not a confounder — a real correction, but a small one, and it must be reported as such
   rather than as "controlling for confounders".
3. **The treated arm is tiny.** 27 discounted lines, 4 SKUs, 70 treated SKU-months. The cluster
   bootstrap reflects this honestly; a line-level interval would not.

**What a real test would need.** To detect a 10% lift in units at 80% power:

| Effect to detect | Δ (units) | n per arm |
| --- | --- | --- |
| 5% | 0.68 | **6,805 SKU-months** |
| 10% | 1.37 | **1,702 SKU-months** |
| 20% | 2.74 | **426 SKU-months** |
| 30% | 4.10 | **190 SKU-months** |

The world supplies **70** treated SKU-months. At the effect sizes a business would act on, this is
two orders of magnitude short — and no amount of clever analysis closes that gap.

## 8. Threats to validity

| Threat | Status |
| --- | --- |
| **Mislabelled treatment** | Checked: zero discounted lines outside treated cells. Asserted as a test |
| **Contaminated panel** | Supplier invoices (USD/CNY costs) were excluded. The first version included them and was wrong |
| **Parallel trends** | Tested, not assumed. The control group moves on mix alone; a placebo DiD on fake treatments bounds the composition artefact |
| **Non-independence** | Handled by clustering the bootstrap on SKU |
| **Multiple comparisons** | Not run — one outcome, one test, one design |
| **Synthetic-world artefacts** | The known truth is used as an estimator check. This is only possible because the world is synthetic, and it is not available in a real setting |
| **Survivorship / selection** | Not applicable: no entity enters or leaves the panel |

## 9. Pre-registered decision rule

| If | Then |
| --- | --- |
| The estimate is within ~10% of the known truth | The design is adequate for the margin question; the mechanical effect is confirmed |
| The DiD is no better than the naive comparison | Say so — the extra machinery is not justified |
| The placebo is a large share of the real effect | The estimate is not interpretable as a treatment effect |
| The demand question cannot be answered | Say so plainly, and give the sample size a real test would need |

## 10. What I would do next

1. **Prospective A/B test on a live price change**, with the sample size from §7 — the only design
   that can answer H2.
2. **Model price elasticity explicitly** by adding a demand response to the generator, so the world
   contains the mechanism the business question depends on.
3. **Pre-period parallel-trends testing** with more SKUs, which the current 4-cluster design cannot
   support.
