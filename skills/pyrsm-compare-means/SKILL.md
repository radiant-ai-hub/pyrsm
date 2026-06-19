---
name: pyrsm-compare-means
description: Compare means of a numeric variable across two or more groups in Python using the pyrsm library's `compare_means` class. Use this skill whenever a student or analyst wants to run a two-sample t-test, a paired t-test, or an ANOVA-style set of pairwise comparisons; switch to a Wilcoxon rank-sum / signed-rank test for skewed or small samples; apply Bonferroni adjustment to control family-wise error across multiple comparisons; choose between independent and paired samples; or work with the pyrsm package for any group-mean-comparison task — even if they don't explicitly say "pyrsm" or "compare_means". Triggers include phrases like "compare means across groups", "t-test between two groups", "ANOVA across professor ranks", "is salary different by sex/discipline", "before-after paired test", "Wilcoxon rank-sum test", "Bonferroni adjustment", "pairwise comparisons with multiple-testing correction", or any mention of comparing a continuous outcome across categorical groups in a marketing/business analytics class.
---

# pyrsm compare-means workflow

This skill walks the user through pairwise mean comparisons across the levels of a grouping variable using the `pyrsm.basics.compare_means` class — from loading their data to translating the test output into plain-English statements about which groups differ, by how much, and whether the differences survive a multiple-testing correction. It is designed for students learning hypothesis testing in a business/marketing analytics context.

For deep reference on the `compare_means` API, the choice between t-test and Wilcoxon, paired vs independent samples, Bonferroni and other adjustments, and how to read pairwise CIs vs the omnibus question, see `references/compare-means.md`.

## Output policy

Two rules apply to **every** pyrsm call you make in this skill:

1. **Show the full output verbatim.** When you call `.summary()`, `.predict()`, or any pyrsm method that prints results, paste the **complete printed output** into your response — every line of the coefficient table, every statistic, every row. Do NOT abbreviate. Do NOT replace it with a high-level "main takeaways" bullet list. Do NOT skip the table to "save space". The user is reading your response because they want to see the actual numbers, not your summary of them.

2. **Provide the full pedagogical interpretation, not lame bullets.** When walking through results, use the detailed templates in `references/<name>.md`: state hypotheses explicitly, walk every coefficient or cell with its real-world units from the sidecar description, give the significance verdict at the 5% level, use the "holding constant" qualifier (where applicable), and tie back to the business question. A response like "X is positive and significant; Y is positive and significant; Z is not" is exactly the failure mode this skill is built to prevent.

3. **Lead with the requested output.** When the user asks for a specific plot, table, or diagnostic (e.g., "create and interpret PDP", "show me the residual dashboard", "generate PIP", "give me the confusion matrix", "plot the gains chart"), your interpretation must focus on THAT output — not on a re-interpretation of the model's coefficient table or summary. Pi's failure pattern is to say *"I'll generate X and then interpret the fitted results"* and then quietly produce a coefficient walk-through. The user already saw the coefficient interpretation in the previous turn; do not repeat it unless they explicitly ask. Lead with the requested output's content, using the dedicated template for that output if the skill has one (Step 9 for PIP, Step 10 for PDP, etc.).

If your context is genuinely constrained and you cannot reproduce the full output, say so explicitly — never silently abbreviate.


---

## Step 1 — Ask for the data location

The very first thing to do (before importing anything, before writing any code) is ask the user where their data lives. Require an **absolute path**, because relative paths are a frequent source of "file not found" frustration for students who are not yet comfortable with the shell.

Example phrasing:

> Before we start, can you give me the **absolute path** to the data file you want to analyze? For example: `/Users/yourname/Downloads/salary.parquet`. Any of `.parquet`, `.csv`, `.tsv`, `.feather`, `.arrow`, `.xlsx`, or `.json` will work.

If the user gives you a relative path, a `~`-style path, or just a filename, ask again for an absolute path rather than guessing. If the path doesn't exist, say so plainly and ask them to double-check.

## Step 2 — Load the data into a polars DataFrame

Use `scripts/load_data.py` to do the loading. It handles file-type detection by extension, reads the file with the appropriate polars reader, and also looks for a sidecar markdown description file in the same folder.

Run it like this:

```bash
python scripts/load_data.py "<absolute-path-to-data>"
```

The script prints a small JSON report describing the loaded dataset (path, shape, column names with dtypes) and the path of any sidecar `.md` it found.

### Sidecar description file

In pyrsm-style course materials, every dataset is shipped alongside a markdown file that documents what the variables mean. The convention is `<dataset>_description.md` next to `<dataset>.parquet` (e.g., `salary.parquet` ↔ `salary_description.md`). The loader script searches for this and several similar patterns automatically.

If a sidecar is found, **read it before proposing what to compare**. The description tells you the units, what the categorical levels mean, which variables are paired vs independent, and whether group sizes are roughly balanced. This is the difference between a generic "Prof earns more than AsstProf" and a useful "Full professors earn $46,000 more on average than assistant professors in this 2008-09 sample (95% CI [$42,118, $49,874])".

If no sidecar is found, say so and ask the user to briefly describe the variables. Don't invent meanings.

## Step 3 — Decide together what to compare

Once the data and (optionally) the description are in hand, propose a test specification and confirm it with the user before running. You need seven things:

- **Grouping variable (`var1`)** — categorical column whose levels define the groups to compare.
- **Numeric variable (`var2`)** — the continuous outcome whose mean is compared across `var1` levels.
- **Pairs to compare (`comb`)** — list of `"level1:level2"` strings. Default `[]` means *all* pairwise combinations. Use this to focus on specific contrasts (e.g., `comb=["Prof:AsstProf"]`).
- **Alternative hypothesis (`alt_hyp`)** — `"two-sided"`, `"greater"`, or `"less"`. For pairwise tests, `"greater"` means `mean(level1) > mean(level2)` for each `"level1:level2"` pair.
- **Confidence level (`conf`)** — almost always `0.95`.
- **Sample type (`sample_type`)** — `"independent"` (default; different subjects in each group) or `"paired"` (same subjects measured twice, e.g., before/after). Paired requires equal group sizes.
- **Adjustment (`adjust`)** — `None` (default, no correction) or `"bonferroni"` (or any other method passed to `statsmodels.stats.multitest.multipletests`). See Step 6.
- **Test type (`test_type`)** — `"t-test"` (default, Welch's t-test for independent or paired t-test) or `"wilcox"` (Wilcoxon rank-sum for independent, Wilcoxon signed-rank for paired). See Step 7.

If the description file or the user's question makes the choice obvious (e.g., "does salary differ across professor ranks?"), state your proposed specification — `var1="rank"`, `var2="salary"`, `comb=[]`, `alt_hyp="two-sided"`, `sample_type="independent"`, `adjust=None`, `test_type="t-test"` — and ask the user to confirm. **Ask about adjustment explicitly**: "you'll have 3 pairwise tests; do you want to apply a Bonferroni correction to control the family-wise error rate?"

Do a quick sanity check before running:

- Is `var2` numeric? It must be. If `var1` is numeric and you pass a *list* of numeric columns as `var2`, the data are auto-melted to long form.
- How many levels does `var1` have? More levels = more pairwise tests = more concern about multiple-testing. Compute `k * (k - 1) / 2`.
- Group sizes — are any too small for a t-test? Welch's t-test is reasonably robust for n ≥ ~15 per group; below that, consider Wilcoxon.
- For paired samples: do the two groups have the same number of rows? They must.

## Step 4 — Run the test

Write a short, runnable Python script (or run it interactively if you have a Python REPL available). Keep it minimal and readable — students will look at this code:

```python
import polars as pl
import pyrsm as rsm

# 1. Load the data (absolute path)
data_path = "<absolute-path>"
df = pl.read_parquet(data_path)   # adjust reader for the actual file type

# 2. Run the comparison
cm = rsm.basics.compare_means(
    {"<dataset_name>": df},        # dict so the dataset name shows up in the summary
    var1="<group>",
    var2="<numeric>",
    comb=[],                       # [] for all pairs, or e.g. ["Prof:AsstProf"]
    alt_hyp="two-sided",           # or "greater" / "less"
    conf=0.95,
    sample_type="independent",     # or "paired"
    adjust=None,                   # or "bonferroni"
    test_type="t-test",            # or "wilcox"
)

# 3. Inspect
cm.summary(extra=True)             # extra=True adds se, t, df, CI columns
```

Notes:
- Pass the data as `{"name": df}` (a dict with one key) rather than just `df`, so the summary header prints a meaningful dataset name instead of `"Not provided"`.
- `pyrsm` accepts polars DataFrames directly.
- `summary(extra=True)` is the default for a first look — it adds se, t-value, df, and CI columns to the pairwise table. `summary()` (without `extra`) prints just `Null hyp.`, `Alt. hyp.`, `diff`, `p.value`, and stars.
- `cm.descriptive_stats` (per-group n, mean, sd, se, me) and `cm.comp_stats` (the pairwise table) are available programmatically.

## Step 5 — Interpret the output

This is where most of the value is. Walk the user through the summary in roughly this order — see `references/compare-means.md` for detailed templates and worked examples:

1. **State what you're doing.** "We're comparing the mean of `<var2>` across the `<k>` levels of `<var1>` using a `<test_type>`. With `<k>` levels there are `<k*(k-1)/2>` pairwise comparisons. `<None | Bonferroni>` adjustment was applied to the p-values."

2. **Walk the descriptive table.** Group means with n, sd, se, me. Note any obvious differences and any very-unequal sample sizes (which `var1` levels are small enough to worry about?).

3. **State the hypotheses for each pair.** From the `Null hyp.` and `Alt. hyp.` columns:
   - H₀: `mean(level1) = mean(level2)`.
   - Hₐ: `mean(level1) <alt_hyp_sign> mean(level2)`.

4. **For each pair, state the test result three equivalent ways.** Same pattern as `single_mean`:
   - **p-value**: "p = `<p>`, `<smaller | not smaller>` than α; we `<reject | fail to reject>` H₀."
   - **Confidence interval**: "the 95% CI for the difference is [`<lo>`, `<hi>`]. Because 0 `<is | is not>` inside, we `<reject | fail to reject>` H₀."
   - **t-value vs critical t**: optional but pedagogically useful — `|t|` = `<t>` on `<df>` df; critical t for the alt_hyp is approximately `<crit>`.

5. **Translate to a business conclusion.** Don't stop at "we reject H₀". Tie back to the units. For the salary-by-rank example: "Assistant professors earn $13,100 less on average than associate professors (95% CI [$9,140, $17,061], p < .001) — a substantial gap of about 14% of associate-prof pay."

6. **If you adjusted, say so explicitly and explain what changed.** Bonferroni multiplies each raw p-value by the number of pairs (capped at 1). A pair that was p = 0.04 with 5 unadjusted tests becomes p = 0.20 adjusted — and goes from "significant" to "not". Always state which p-values are adjusted.

### Don't cherry-pick significant pairs — and don't pretend the omnibus question is the same as the pairwise one

The classic compare-means failure modes:

- **Cherry-picking after the fact.** Running pairwise tests, reading off whichever happen to be significant, and reporting only those. With `k` levels and α = 0.05, you'd expect ~5% of zero-difference pairs to look significant by chance. The right protocol is to decide which pairs you care about *before* looking, or to apply a multiple-testing correction.
- **Confusing the omnibus and pairwise questions.** A pairwise approach answers "does Prof differ from AsstProf?" An omnibus ANOVA-style question would be "do means differ *somewhere* across the levels?". The two questions have different p-values and different interpretations. Pyrsm's `compare_means` is explicitly pairwise; if the user wants an omnibus F-test, mention that they can fit `rsm.model.regress` with `var1` as a categorical predictor and read off the F-statistic.
- **One-sided pairwise after seeing the data.** Setting `alt_hyp="less"` because the sample showed `level1 < level2` doubles the effective false-positive rate. Pick `alt_hyp` from the research question before looking, or stay two-sided.
- **Reading significance with very unequal group sizes.** Welch's t-test is robust to unequal variances but not to extreme group imbalance. If one level has n = 5 and another has n = 500, the test will still report a p-value, but the small-group standard error dominates and the effective sample size for inference is much smaller than n = 505.

So: **always decide which pairs matter before looking, and apply Bonferroni (or equivalent) when you genuinely look at all of them.**

## Step 6 — Multiple-testing adjustment (the critical concept)

This is the **defining pedagogical concept** for `compare_means`. When you run multiple pairwise tests at α = 0.05, the *family-wise* error rate — the probability of at least one false positive when all H₀s are true — grows with the number of tests:

| Number of independent tests | Family-wise error if α = 0.05 each |
|---|---|
| 1 | 0.05 |
| 3 | ≈ 0.14 |
| 5 | ≈ 0.23 |
| 10 | ≈ 0.40 |
| 20 | ≈ 0.64 |

So with 10 pairwise tests at α = 0.05, there's a 40% chance of at least one false positive even if all true differences are zero. Reporting "we found 1 significant pair out of 10" without correction is roughly as informative as a coin flip.

### Bonferroni correction

`compare_means` exposes `adjust="bonferroni"`, which multiplies each raw p-value by the number of pairs (capping at 1.0). Equivalent to requiring `p < α/k` for each individual test, where `k` is the number of pairs. Conservative but easy to explain.

Other methods (`"holm"`, `"hochberg"`, `"fdr_bh"`, etc.) are passed through to `statsmodels.stats.multitest.multipletests`. Bonferroni is the safest default for class work.

### When you can skip adjustment

- You only care about *one specific* pre-specified pair. Set `comb=["that_pair"]` and run a single test — no correction needed.
- The omnibus question is "do means differ somewhere?". Use the regression-based F-test instead.
- All comparisons are highly significant (p < .001) with a tight effect. Adjustment doesn't change the substantive conclusion, but mention it for completeness.

### When you should *always* adjust

- You ran more than one comparison and want to claim individual significance on any of them.
- You ran exploratory comparisons across many group/outcome combinations and are reporting whichever came back significant.
- Reviewers, advisors, or the analytics standards in your course require it.

### How Bonferroni interacts with `alt_hyp`

In `compare_means`, when `adjust="bonferroni"` is combined with a one-sided `alt_hyp`, the implementation doubles the working alpha before passing it to `multipletests` to keep the comparison framework consistent. The reported p-values are then adjusted within that framework. Practically: just use the adjusted p-values as printed; don't try to re-derive them by hand.

## Step 7 — t-test vs Wilcoxon

`compare_means` supports two test types:

- **`test_type="t-test"`** (default) — Welch's t-test for independent samples (does not assume equal variances), paired t-test for paired samples. Assumes the data are approximately normally distributed *or* the sample is large enough for the CLT to kick in. Robust for n ≥ 15–30 per group even with moderate non-normality.
- **`test_type="wilcox"`** — Wilcoxon rank-sum (Mann–Whitney U) for independent samples, Wilcoxon signed-rank for paired samples. Tests the null hypothesis that the *distributions* are the same (loosely interpretable as "the medians are equal", with caveats). Robust to outliers and skew; doesn't require normality.

### When to use which

- **Large samples (n ≥ 30 per group), roughly symmetric distributions** → t-test.
- **Small samples (n < 15 per group) and you don't know the distribution** → Wilcoxon.
- **Heavy skew, outliers, or ordinal-ish data** → Wilcoxon.
- **Distributions look weird and you want a robustness check** → run both and compare. If they agree, the t-test conclusion is robust; if they disagree, the Wilcoxon is more trustworthy.

### Interpretive note for Wilcoxon

The Wilcoxon test compares stochastic equality of distributions, *not* the mean per se. With non-normal data, the printed `diff` (which is the difference of means) doesn't quite match what the Wilcoxon p-value tests. State the Wilcoxon conclusion in terms of "the distributions of `<var2>` differ across groups" rather than "the means differ", unless you're prepared to defend a stronger location-shift interpretation.

## Step 8 — Independent vs paired samples

Set `sample_type="paired"` when:

- The same subjects are measured twice (before-after, control-treatment-within-subject).
- Each row of `var1=level1` is naturally paired with a corresponding row of `var1=level2` (matched twins, repeated measures, paired surveys).

Set `sample_type="independent"` (default) when:

- Different subjects in each group.
- No natural pairing.

Paired tests have more power than independent tests on the same data, because they remove between-subject variance. But misusing paired on truly independent data inflates the false-positive rate; misusing independent on truly paired data wastes power. Get this right.

**Paired requires equal group sizes** (the code raises if `len(x) != len(y)`). In the long-form data shape pyrsm uses, that means each level has the same number of rows.

## Step 9 — Plot (offer, don't force)

After the basic interpretation, **offer** a visualization:

```python
cm.plot(plots="scatter")   # default; jittered scatter with group means
cm.plot(plots="box")       # boxplots per group
cm.plot(plots="density")   # density curves overlaid
cm.plot(plots="bar")       # bar chart of group means
```

All four return plotnine ggplot objects.

- **Scatter** (default): every observation as a point + a horizontal crossbar at the group mean. Best for spotting outliers and getting a feel for spread.
- **Box**: classic boxplot. Best for comparing medians and IQRs across groups.
- **Density**: smoothed distributions overlaid. Best for spotting bimodality or seeing how distributions differ in shape.
- **Bar**: bar chart of means only. Loses spread information; useful for a quick summary slide but inferior to scatter or box for analysis.

## Step 10 — Probability-calculator companion (optional)

For a specific pair, you can compute the critical t value:

```python
# Two-sided critical t at alpha=0.05, df = (look up from cm.comp_stats)
pc = rsm.basics.prob_calc("tdist", df=<df>, pub=0.975)
pc.summary()
```

For a Bonferroni-adjusted critical t with k tests, use `pub=1 - (alpha/(2*k))` (for two-sided).

If the user wants more on `prob_calc`, route them to the `pyrsm-prob-calc` skill.

## Step 11 — Related tests (when to switch)

`compare_means` is the right tool when you have one continuous outcome and one categorical grouping variable. If the situation is different, suggest:

- One sample, comparing its mean to a fixed benchmark → `single_mean`.
- Comparing *proportions* (binary outcomes) across groups → `compare_props`.
- Continuous outcome, multiple categorical predictors and/or covariates, omnibus F-test → `regress` (fit with the categorical as a predictor; the F-test answers "is *any* level different?").
- Two categorical variables, association testing → `cross_tabs`.

Don't switch silently — explain why the alternative is a better fit and confirm with the user before changing tools.

---

## Style notes

- **Pedagogical tone.** Always state the null and alternative hypotheses explicitly per pair. Always tie significance back to a 5% threshold. Always interpret `diff` in the original units, not just as a p-value.
- **Use the units from the description file.** "AsstProf < Prof" is much weaker than "assistant professors earn ~$46,000 less than full professors on average".
- **Multiple-testing adjustment.** This is the unique critical concept for compare_means. Always ask "how many pairs are you going to look at?" and offer Bonferroni when the count is more than 1–2 and the user hasn't already chosen.
- **Pick `alt_hyp`, pairs, and adjust before looking at the data.** Doing it after is p-hacking.
- **Welch's t-test is the default for `test_type="t-test"`.** It does NOT assume equal variances, which is one less assumption to defend.
- **Wilcoxon for small / skewed / outlier-heavy samples.** Run both and compare if you're not sure.
- **Don't over-engineer.** Plain top-level code that the student can copy line by line.
- **One step at a time.** After Step 4 (run) wait for the user to look at the tables before barreling into plots or adjustment experiments.

## Growing this skill

This skill is one of a family of `pyrsm-basics-*` skills covering the `pyrsm.basics` module. When `compare_means` isn't the right tool, point the user to the sibling skill named for the analysis they need (`pyrsm-single-mean`, `pyrsm-compare-props`, `pyrsm-regress`, `pyrsm-prob-calc`, etc.). The workflow shape — ask for absolute path → load via `scripts/load_data.py` → read sidecar → propose spec → run → interpret — is shared across the family.
