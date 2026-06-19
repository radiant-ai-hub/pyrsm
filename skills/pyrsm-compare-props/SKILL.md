---
name: pyrsm-compare-props
description: Compare proportions of a categorical outcome across two or more groups in Python using the pyrsm library's `compare_props` class. Use this skill whenever a student or analyst wants to test if the proportion of a "success" level (yes / churned / clicked / survived) differs across groups defined by another categorical variable, focus on specific pair comparisons via `comb`, apply Bonferroni adjustment, validate the chi-squared cell-count assumption, or work with the pyrsm package for any group-proportion-comparison task — even if they don't explicitly say "pyrsm" or "compare_props". Triggers include phrases like "compare proportions across groups", "does survival rate differ by class", "did the click-rate change between A and B and C", "two-proportion z-test", "is the yes-rate the same for males and females", "pairwise proportion comparisons with adjustment", or any mention of comparing a binary outcome across two or more categorical groups in a marketing/business analytics class.
---

# pyrsm compare-props workflow

This skill walks the user through pairwise proportion comparisons across the levels of a grouping variable using the `pyrsm.basics.compare_props` class — from loading their data to translating the test output into plain-English statements about which groups' rates of the "success" outcome differ, by how many percentage points, and whether the differences survive a multiple-testing correction. It is designed for students learning hypothesis testing in a business/marketing analytics context.

For deep reference on the `compare_props` API, the importance of the `lev` choice, the chi-squared cell-count assumption, Wald CIs, and `comb` for focused contrasts, see `references/compare-props.md`.

## Output policy

Two rules apply to **every** pyrsm call you make in this skill:

1. **Show the full output verbatim.** When you call `.summary()`, `.predict()`, or any pyrsm method that prints results, paste the **complete printed output** into your response — every line of the coefficient table, every statistic, every row. Do NOT abbreviate. Do NOT replace it with a high-level "main takeaways" bullet list. Do NOT skip the table to "save space". The user is reading your response because they want to see the actual numbers, not your summary of them.

2. **Provide the full pedagogical interpretation, not lame bullets.** When walking through results, use the detailed templates in `references/<name>.md`: state hypotheses explicitly, walk every coefficient or cell with its real-world units from the sidecar description, give the significance verdict at the 5% level, use the "holding constant" qualifier (where applicable), and tie back to the business question. A response like "X is positive and significant; Y is positive and significant; Z is not" is exactly the failure mode this skill is built to prevent.

3. **Lead with the requested output.** When the user asks for a specific plot, table, or diagnostic (e.g., "create and interpret PDP", "show me the residual dashboard", "generate PIP", "give me the confusion matrix", "plot the gains chart"), your interpretation must focus on THAT output — not on a re-interpretation of the model's coefficient table or summary. Pi's failure pattern is to say *"I'll generate X and then interpret the fitted results"* and then quietly produce a coefficient walk-through. The user already saw the coefficient interpretation in the previous turn; do not repeat it unless they explicitly ask. Lead with the requested output's content, using the dedicated template for that output if the skill has one (Step 9 for PIP, Step 10 for PDP, etc.).

If your context is genuinely constrained and you cannot reproduce the full output, say so explicitly — never silently abbreviate.


---

## Step 1 — Ask for the data location

The very first thing to do (before importing anything, before writing any code) is ask the user where their data lives. Require an **absolute path**.

Example phrasing:

> Before we start, can you give me the **absolute path** to the data file you want to analyze? For example: `/Users/yourname/Downloads/titanic.parquet`. Any of `.parquet`, `.csv`, `.tsv`, `.feather`, `.arrow`, `.xlsx`, or `.json` will work.

If the user gives you a relative path, a `~`-style path, or just a filename, ask again for an absolute path rather than guessing.

## Step 2 — Load the data into a polars DataFrame

Use `scripts/load_data.py` to do the loading:

```bash
python scripts/load_data.py "<absolute-path-to-data>"
```

The script prints a small JSON report describing the loaded dataset and the path of any sidecar `.md` it found.

### Sidecar description file

If a sidecar is found, **read it before proposing what to compare**. The description tells you which levels of the response are the substantive "success" (e.g., `survived="Yes"`, `churn="yes"`, `click="1"`), how the groups in `var1` are coded, and any quirks. For the titanic example, the sidecar tells you `survived` has levels `"Yes"`/`"No"` and `pclass` has `"1st"`/`"2nd"`/`"3rd"` — important for picking `lev` correctly.

If no sidecar is found, say so and ask the user to briefly describe the variables. Don't invent meanings.

## Step 3 — Decide together what to test

Once the data and (optionally) the description are in hand, propose a test specification and confirm it with the user before running. You need seven things:

- **Grouping variable (`var1`)** — categorical column whose levels define the groups being compared.
- **Response variable (`var2`)** — categorical column with at least two levels; one will count as "success".
- **Success level (`lev`)** — the level of `var2` that counts as "success" in each group. This is a **substantive choice** tied to the research question, not a default. Ask: "which outcome are we measuring the rate of?" For titanic survival, `lev="Yes"` measures the survival rate; `lev="No"` measures the death rate. The two reframings give numerically related but conceptually distinct interpretations — and the printed `diff` direction will flip.
- **Pairs to compare (`comb`)** — list of `"level1:level2"` strings of `var1`. Default `[]` means all pairwise combinations. Use `comb=["1st:2nd", "1st:3rd"]` to focus on specific contrasts.
- **Alternative hypothesis (`alt_hyp`)** — `"two-sided"`, `"greater"`, or `"less"`. Pick from the research question before looking at the data.
- **Confidence level (`conf`)** — almost always `0.95`.
- **Adjustment (`adjust`)** — `None` (default) or `"bonferroni"` (or any other statsmodels.stats.multitest method). See Step 6.

If the description file or the user's question makes the choice obvious (e.g., "did survival differ by passenger class?"), state your proposed specification — `var1="pclass"`, `var2="survived"`, `lev="Yes"`, `comb=[]` (all 3 pairs), `alt_hyp="two-sided"`, `adjust="bonferroni"` recommended for 3 pairs — and ask the user to confirm.

Do a quick sanity check before running:

- Is `lev` actually a level of `var2`? Typos (`"yes"` vs `"Yes"`) silently produce `ns = 0` and nonsensical results. Check `df[var2].value_counts()` if unsure.
- How many levels does `var1` have? More levels = more pairwise tests = more concern about multiple-testing.
- **The chi-squared expected-cell assumption.** For each pair, look at the smallest expected cell count: `min(n_1 * pooled_p, n_1 * (1 - pooled_p), n_2 * pooled_p, n_2 * (1 - pooled_p))` where `pooled_p = (c_1 + c_2) / (n_1 + n_2)`. If any of these is < ~5, the chi-squared / z-test approximation is unreliable. See Step 7.

## Step 4 — Run the test

Write a short, runnable Python script (or run it interactively if you have a Python REPL available):

```python
import polars as pl
import pyrsm as rsm

# 1. Load the data (absolute path)
data_path = "<absolute-path>"
df = pl.read_parquet(data_path)   # adjust reader for the actual file type

# 2. Run the comparison
cp = rsm.basics.compare_props(
    {"<dataset_name>": df},        # dict so the dataset name shows up in the summary
    var1="<group>",
    var2="<response>",
    lev="<success-level>",
    comb=[],                       # [] for all pairs, or e.g. ["1st:2nd", "1st:3rd"]
    alt_hyp="two-sided",
    conf=0.95,
    adjust=None,                   # or "bonferroni"
)

# 3. Inspect
cp.summary(extra=True)             # extra=True adds chisq.value, df, CI columns
```

Notes:
- Pass the data as `{"name": df}` (a dict with one key) rather than just `df`, so the summary header prints a meaningful dataset name instead of `"Not provided"`.
- `pyrsm` accepts polars DataFrames directly.
- `summary(extra=True)` is the default for a first look — it adds `chisq.value`, `df`, and CI columns. `summary()` (without `extra`) prints just `Null hyp.`, `Alt. hyp.`, `diff`, `p.value`, and stars.
- `cp.descriptive_stats` (per-group `ns`, `p`, `n`, `sd`, `se`, `me`) and `cp.comp_stats` (pairwise table) are available programmatically.
- Test statistic is a z-statistic from `statsmodels.stats.proportion.proportions_ztest`; the printed `chisq.value` is `z**2`. P-value comes from the z-test. CI is the Wald interval for the difference of proportions.

## Step 5 — Interpret the output

Walk the user through the summary in roughly this order — see `references/compare-props.md` for detailed templates and worked examples:

1. **State what you're doing.** "We're comparing the proportion of `<lev>` in `<var2>` across the `<k>` levels of `<var1>`. With `<k>` levels there are `<k*(k-1)/2>` pairwise comparisons. Adjustment: `<None | Bonferroni>`."

2. **Walk the descriptive table.** Per-group `ns` (successes), `p` (proportion of successes), `n` (non-missing observations). Highlight the rank order: which group has the highest proportion, which the lowest.

3. **State the hypotheses for each pair.** From the `Null hyp.` and `Alt. hyp.` columns:
   - H₀: `p(level1) = p(level2)`.
   - Hₐ: `p(level1) <alt_hyp_sign> p(level2)`.

4. **For each pair, state the test result.** Three equivalent ways:
   - **p-value**: "p = `<p>`, `<smaller | not smaller>` than α; we `<reject | fail to reject>` H₀."
   - **Confidence interval**: "the 95% CI for the difference in proportions is [`<lo>`, `<hi>`]. Because 0 `<is | is not>` inside, we `<reject | fail to reject>` H₀."
   - **Chi-squared / critical-value comparison**: chi-sq = `<value>` on 1 df. Critical chi-squared at α = 0.05 is 3.841. Reject if `chisq > 3.841`.

5. **Translate to a business conclusion.** Don't stop at "we reject H₀". Tie back to percentage points and the business meaning. For the titanic example: "63.5% of 1st-class passengers survived versus 44.1% of 2nd-class — a 19.4 percentage-point gap (95% CI [11.2 pp, 27.7 pp], p < .001). Socio-economic class was strongly associated with survival on the Titanic."

6. **If you adjusted, say so explicitly.** Bonferroni-adjusted p-values are larger than raw. Always state which p-values are adjusted.

### Don't cherry-pick pairs — and don't pick `lev` after looking at the data

The classic compare-props failure modes:

- **Cherry-picking significant pairs.** With `k*(k-1)/2` pairs run unadjusted, the family-wise false-positive rate inflates fast. Always pre-specify pairs (`comb=[...]`) or apply Bonferroni.
- **Picking `lev` based on which side gives a "more significant" result.** `lev="Yes"` and `lev="No"` give the *same* p-value (the z-statistic flips sign but `z**2` is identical). But the *direction* and *substantive framing* changes. Pick `lev` from the research question, not the result.
- **One-sided pairwise after seeing direction.** Setting `alt_hyp="greater"` because group A's sample proportion is higher than group B's doubles the effective false-positive rate. Pick direction from the question.
- **Violated chi-squared assumption ignored.** With small groups and a rare outcome, the underlying z-test approximation isn't reliable. Pyrsm will compute and print a p-value regardless — but it may be misleading. Check the expected cell counts. See Step 7.

## Step 6 — Multiple-testing adjustment

Same conceptual concern as in `compare_means`. With `k` levels you have `k*(k-1)/2` pairwise tests; the family-wise error rate inflates with the number of tests.

| Number of pairwise tests | FWER at α = 0.05 |
| --- | --- |
| 1 | 0.05 |
| 3 | 0.143 |
| 6 (k=4) | 0.265 |
| 10 (k=5) | 0.401 |

### Bonferroni in pyrsm

Set `adjust="bonferroni"`. Each raw p-value is multiplied by the number of pairs (capped at 1). Equivalent to requiring `p < α/k`. Conservative but defensible. The displayed p-values are the adjusted ones.

Other statsmodels methods (`"holm"`, `"fdr_bh"`, etc.) are passed through. Bonferroni is the safe class default.

### When to skip adjustment

- A single pre-specified pair: `comb=["that_pair"]`, no correction needed.
- The omnibus question is "does the proportion differ somewhere?". Use `cross_tabs` (chi-squared test of independence) instead.

### Always adjust when

- You're running multiple comparisons and want to claim individual significance.
- You're reporting "the pairs that came back significant" from a fishing expedition.

## Step 7 — The chi-squared cell-count assumption

This is the **defining critical concept** for `compare_props`.

The two-proportion z-test (and equivalent chi-squared test) approximates a discrete sampling distribution with a normal. The approximation requires enough expected counts in *every* 2×2 cell of the implicit contingency table for the pair.

### The rule of thumb

For each pair `(var1=v1, var1=v2)`, build the implicit 2×2 table:

|        | success | failure |
| ------ | ------- | ------- |
| `v1`   | `c1`    | `n1 - c1` |
| `v2`   | `c2`    | `n2 - c2` |

Compute the pooled proportion `p_pool = (c1 + c2) / (n1 + n2)` and the **expected** cell counts under H₀:

- `e1_success = n1 * p_pool`
- `e1_failure = n1 * (1 - p_pool)`
- `e2_success = n2 * p_pool`
- `e2_failure = n2 * (1 - p_pool)`

**All four should be at least 5.** If any is < 5, the z-test / chi-squared p-value is biased.

### Pyrsm's behavior

`compare_props` does NOT check this assumption — it will happily compute a p-value and CI for any cell counts. The skill should check manually.

### When violated

- **Collapse categories** in `var1` or `var2` to bring small cells up. For 4-level groups with one tiny level, merging the tiny level with an adjacent one is principled.
- **Use Fisher's exact test** instead. Not exposed by pyrsm; compute via `scipy.stats.fisher_exact` on a 2×2 sub-table.
- **Get more data.** Sometimes feasible.

### When to mention this proactively

- The user has a small group (n < 30) and/or a rare outcome (p < 5%).
- A 2×2 inspection shows any cell < 5.
- Reviewers / advisors / course standards require it.

## Step 8 — Plot (offer, don't force)

After the basic interpretation, **offer** a visualization:

```python
cp.plot(plots="bar")     # bar chart of proportions per group
cp.plot(plots="dodge")   # side-by-side bars for each level of var2 within each var1 group
```

- **Bar** (default): one bar per group, height = proportion of `lev`. Quick comparison of rates.
- **Dodge**: stacked / side-by-side bars showing all levels of `var2` per `var1` group. Useful when `var2` has more than two levels or when you want to see the full breakdown.

Both return plotnine ggplot objects.

## Step 9 — Probability-calculator companion (optional)

To find the critical chi-squared value:

```python
pc = rsm.basics.prob_calc("chisq", df=1, pub=0.95)
pc.summary()
# Critical chi-squared at alpha=0.05 with 1 df is 3.841
```

Or the critical z:

```python
pc = rsm.basics.prob_calc("norm", mean=0, sd=1, pub=0.975)   # two-sided alpha=0.05
pc.summary()
# Critical z is ±1.96
```

If the user wants more on `prob_calc`, route them to the `pyrsm-prob-calc` skill.

## Step 10 — Related tests (when to switch)

`compare_props` is the right tool when you have a binary categorical outcome and a categorical grouping variable. If the situation is different, suggest:

- One group, comparing its proportion to a fixed benchmark → `single_prop`.
- Two categorical variables both multi-level, omnibus association → `cross_tabs`.
- One categorical variable's distribution against an expected one → `goodness`.
- Continuous outcome, multiple groups → `compare_means`.
- Modeling how the probability of success depends on covariates → `logistic`.

Don't switch silently — explain why the alternative is a better fit and confirm with the user before changing tools.

---

## Style notes

- **Pedagogical tone.** Always state H₀/Hₐ explicitly per pair. Always tie significance back to a 5% threshold. Always interpret `diff` in percentage points, not just as a p-value.
- **Use the units from the description file.** "1st > 3rd" is much weaker than "1st-class passengers survived at 63.5% versus 26.2% for 3rd class — a 37.3 percentage-point gap".
- **Multiple-testing adjustment.** Default to Bonferroni for any non-trivial number of pairs. Pre-specify pairs via `comb=` when the research question is focused.
- **Check the chi-squared cell-count assumption.** This is the critical concept for compare_props. Look at the smallest expected cell count before trusting the p-value.
- **Pick `lev`, `alt_hyp`, and pairs from the question, not the data.** Doing it after is p-hacking.
- **Don't over-engineer.** Plain top-level code that the student can copy line by line.
- **One step at a time.** After Step 4 wait for the user to look at the tables before barreling into plots or adjustment experiments.

## Growing this skill

This skill is one of a family of `pyrsm-basics-*` skills covering the `pyrsm.basics` module. When `compare_props` isn't the right tool, point the user to the sibling skill named for the analysis they need (`pyrsm-single-prop`, `pyrsm-compare-means`, `pyrsm-cross-tabs`, `pyrsm-goodness`, `pyrsm-prob-calc`, etc.). The workflow shape — ask for absolute path → load via `scripts/load_data.py` → read sidecar → propose spec → run → interpret — is shared across the family.
