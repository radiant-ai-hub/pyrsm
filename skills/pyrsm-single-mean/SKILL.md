---
name: pyrsm-single-mean
description: Run and interpret one-sample mean tests in Python using the pyrsm library's `single_mean` class. Use this skill whenever a student or analyst wants to compare a sample mean to a hypothesized population value, decide whether the difference is statistically meaningful, choose a one-sided vs two-sided alternative, build or read a confidence interval around a mean, or work with the pyrsm package for any one-sample-t-test task — even if they don't explicitly say "pyrsm" or "single_mean". Triggers include phrases like "is the average really X?", "test whether the mean equals/exceeds/is below some target", "one-sample t-test", "compare a sample mean to a benchmark", "does this column's mean differ from 1750", "build a 95% CI for the mean", or any mention of testing a single continuous variable against a target value in a marketing/business analytics class.
---

# pyrsm single-mean workflow

This skill walks the user through a complete one-sample-mean hypothesis test using the `pyrsm.basics.single_mean` class — from loading their data to translating the test output into a plain-English decision their manager would understand. It is designed for students learning hypothesis testing in a business/marketing analytics context, so default to clear, pedagogical explanations rather than terse statistical output.

For deep reference on the `single_mean` API, sampling-distribution intuition, alt-hypothesis selection, and the three equivalent ways to evaluate the test (p-value, confidence interval, critical t-value), see `references/single-mean.md`. Read it whenever the user asks a question that goes beyond the basic fit-and-summary loop.

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

> Before we start, can you give me the **absolute path** to the data file you want to analyze? For example: `/Users/yourname/Downloads/demand_uk.parquet`. Any of `.parquet`, `.csv`, `.tsv`, `.feather`, `.arrow`, `.xlsx`, or `.json` will work.

If the user gives you a relative path, a `~`-style path, or just a filename, ask again for an absolute path rather than guessing. If the path doesn't exist, say so plainly and ask them to double-check.

## Step 2 — Load the data into a polars DataFrame

Use `scripts/load_data.py` to do the loading. It handles file-type detection by extension, reads the file with the appropriate polars reader, and also looks for a sidecar markdown description file in the same folder.

Run it like this:

```bash
python scripts/load_data.py "<absolute-path-to-data>"
```

The script prints a small JSON report describing the loaded dataset (path, shape, column names with dtypes) and the path of any sidecar `.md` it found. Use that to plan your next step. After this initial probe, in the actual analysis code, you will do the equivalent inside the Python session you'll run the test in — see Step 4.

### Sidecar description file

In pyrsm-style course materials, every dataset is shipped alongside a markdown file that documents what the variables mean. The convention is `<dataset>_description.md` next to `<dataset>.parquet` (e.g., `demand_uk.parquet` ↔ `demand_uk_description.md`), but other "very similar" filenames are also common in the wild. The loader script searches for these patterns automatically.

If a sidecar is found, **read it before proposing what to test**. The description tells you the units (e.g., "demand is total category sales in units per store per month"), what the variable measures, and any quirks. This is the difference between a generic "the mean is significantly larger" and a useful "average monthly demand is 1953 units per store, significantly above the 1750-unit threshold management cares about".

If no sidecar is found, say so and ask the user to briefly describe the variable. Don't invent meanings.

## Step 3 — Decide together what to test

Once the data and (optionally) the description are in hand, propose a test specification and confirm it with the user before running. You need four things:

- **Variable (`var`)** — one column. Must be numeric.
- **Comparison value (`comp_value`)** — the population mean under H₀. This is a substantive choice, not a statistical one: it should come from the business context (a target, a benchmark, a contract minimum, a prior published estimate). **Ask where the comparison value comes from** if the user doesn't volunteer it.
- **Alternative hypothesis (`alt_hyp`)** — one of `"two-sided"`, `"greater"`, `"less"`. Pick *before* looking at the data, based on the decision being made:
  - `"greater"` if the decision only triggers when the mean is *above* `comp_value` (e.g., "enter the market if demand > 1750").
  - `"less"` if the decision only triggers when the mean is *below* `comp_value` (e.g., "stop the trial if defect rate is below 5%").
  - `"two-sided"` if either direction matters (e.g., "is the new process *different* from the old benchmark?").
- **Confidence level (`conf`)** — almost always `0.95` for a class. Increase to `0.99` only if the user explicitly asks for a stricter standard.

If the description file or the user's question makes the choice obvious (e.g., "is monthly demand above 1750 units?"), state your proposed specification — `var="demand"`, `comp_value=1750`, `alt_hyp="greater"`, `conf=0.95` — and ask the user to confirm.

Do a quick sanity check before running:

- Is `var` numeric? If it's binary 0/1, mention that `single_prop` might fit better, but proceed with `single_mean` if the user wants it (the math still works for proportions of 0s and 1s, it just isn't the standard tool).
- Does `var` have many missing values? `single_mean` drops them automatically (it shows them as `n_missing` in the summary), but flag it so the user knows their effective sample size.
- Is the comp_value zero by accident? `single_mean` defaults `comp_value=0`. If the user is just clicking through and the test is "is the mean different from zero?" — confirm that's what they actually want.

## Step 4 — Run the test

Write a short, runnable Python script (or run it interactively if you have a Python REPL available). Keep it minimal and readable — students will look at this code:

```python
import polars as pl
import pyrsm as rsm

# 1. Load the data (absolute path)
data_path = "<absolute-path>"
df = pl.read_parquet(data_path)   # adjust reader for the actual file type

# 2. Run the one-sample mean test
sm = rsm.basics.single_mean(
    {"<dataset_name>": df},        # dict so the dataset name shows up in the summary
    var="<column>",
    alt_hyp="<two-sided|greater|less>",
    conf=0.95,
    comp_value=<benchmark>,
)

# 3. Inspect
sm.summary()
```

Notes:
- Pass the data as `{"name": df}` (a dict with one key) rather than just `df`, so the summary header prints a meaningful dataset name instead of `"Not provided"`.
- `pyrsm` accepts polars DataFrames directly — no need to convert to pandas first.
- The summary prints two tables: descriptive statistics (mean, n, n_missing, sd, se, me) on top, and the hypothesis-test row (diff, se, t.value, p.value, df, CI bounds, significance stars) on the bottom.

## Step 5 — Interpret the output

This is where most of the value is. Walk the user through the summary in roughly this order — see `references/single-mean.md` for the detailed templates and worked examples:

1. **State the hypotheses in words.** Read them straight off the summary header:
   - H₀: the mean of `<var>` is equal to `<comp_value>`.
   - Hₐ: the mean of `<var>` is `<greater than | less than | not equal to>` `<comp_value>`.
   The header already prints these — repeat them in the user's vocabulary using the units from the sidecar.

2. **Describe the sample.** One sentence: "the sample mean is `<mean>` `<units>` (n=`<n>`, sd=`<sd>`, se=`<se>`)".

3. **State the test result three equivalent ways.** This is the pedagogical core of a single-mean test — every introductory stats class asks students to read the same conclusion through three lenses, and the lenses must agree:
   - **p-value**: "p = `<p.value>`, which is `<smaller | not smaller>` than 0.05, so we `<reject | fail to reject>` H₀."
   - **Confidence interval**: "the 95% CI for the mean is [`<lo>`, `<hi>`]. Because `<comp_value>` `<is | is not>` inside this interval, we `<reject | fail to reject>` H₀." (For a one-sided test the interval is half-open: `(-inf, hi]` for `"less"`, `[lo, +inf)` for `"greater"`.)
   - **t-value vs critical t**: "the observed t-statistic is `<t.value>` on `<df>` df. The critical t for a `<one-sided | two-sided>` test at α=0.05 is approximately `<crit-t>`. Because `|t|` `<exceeds | does not exceed>` the critical value, we `<reject | fail to reject>` H₀."

4. **Translate to a business conclusion.** Don't stop at "we reject H₀". Tie it back to the original question, with units. For the demand-UK example: "the data give strong evidence that average monthly demand per store in the UK exceeds 1750 units (sample mean 1953, 95% lower confidence bound 1897), so management should enter the market."

5. **Effect size.** Always report **how far** the sample mean is from the comparison value, not just whether the difference is significant. The `diff` row in the summary is exactly this. A statistically significant difference that is too small to matter for the business decision is a real possibility, especially with large samples.

### Don't read significance off the p-value alone

After looking at the summary, students very often jump straight from "p < 0.05" to "the result is real and meaningful" — or, in the other direction, from "p = 0.06" to "the mean isn't different". Both moves are wrong, and the skill should push back on them.

The reasons, stated for the student:

- **A p-value just under or just over 0.05 is not a hard cliff.** p = 0.04 and p = 0.06 are nearly identical pieces of evidence. The 0.05 threshold is a convention, not a physical law. A study that finds p = 0.06 has not "shown there is no effect" — it has shown the evidence is suggestive but not conclusive at the conventional bar.
- **A significant p-value is not the same as a large effect.** With n in the thousands, even a tiny `diff` can land at p < .001. Report `diff` (in the original units), the margin of error, and — if possible — whether that effect is large enough to matter for the decision the user is actually trying to make.
- **The confidence interval contains strictly more information than the p-value.** It tells you both *whether* the comparison value is plausibly the true mean (the test) *and* a range of values that are equally consistent with the data. Lead with the CI when explaining the result.

So: **when interpreting, do not lean on the p-value alone**. Walk all three views (p-value, CI, t-vs-critical), discuss the magnitude of `diff`, and only then state the conclusion. If the student wants to dichotomize on p = 0.04 vs p = 0.06, explain why the binary "reject / don't reject" decision is a deliberate simplification and not a statement about underlying reality.

## Step 6 — Plot (offer, don't force)

After the basic interpretation, **offer** to run the histogram diagnostic. The default plot is enough for a one-sample test — there is no need to flood the user with diagnostics.

```python
sm.plot("hist")
```

This shows:
- A histogram of `<var>` (the sample distribution).
- A solid black vertical line at the sample mean.
- Dashed black vertical lines at the confidence-interval bounds.
- A solid red vertical line at the comparison value.

The visual question is simple: **does the red line (comp_value) fall inside the dashed-black confidence interval?**

- If no → reject H₀.
- If yes → fail to reject H₀.

The histogram itself also lets the student eyeball the shape of the sample. If it's badly skewed or has long tails, mention that the t-test is robust for large n thanks to the CLT but may be unreliable on small skewed samples (n < ~30 with strong skew) — and that a non-parametric alternative (Wilcoxon signed-rank, available via `compare_means` with `test_type="wilcox"` on a single group) is worth considering in that case.

## Step 7 — Probability-calculator companion (optional)

Many class assignments ask the student to find the **critical t-value** for the test using a probability calculator. The pyrsm package ships one — `pyrsm.basics.prob_calc("tdist", ...)`:

```python
# critical t for a one-sided "greater" test at alpha = 0.05 with df = 571
pc = rsm.basics.prob_calc("tdist", df=571, pub=0.95)
pc.summary()
```

`pub=0.95` asks for the t-value below which 95% of the probability mass lies (i.e., the upper 5% rejection region for a one-sided "greater" test). For a two-sided test at α=0.05, use `pub=0.975` (the upper 2.5%).

Use this:
- to corroborate the third equivalence in Step 5 ("does the observed t exceed the critical t?")
- when the student is preparing a writeup that asks for the critical value explicitly.

If the user wants more on `prob_calc`, route them to the `pyrsm-prob-calc` skill.

## Step 8 — Related tests (when to switch)

`single_mean` is the right tool when the response is a continuous numeric variable and you have one group. If the situation is different, suggest:

- Response is a 0/1 or yes/no proportion → `single_prop`.
- Comparing the mean of one column across two or more groups → `compare_means`.
- Paired (before/after) measurements on the same subjects → `compare_means` with `sample_type="paired"`.
- Want to test whether a categorical column follows an expected distribution → `goodness`.
- Want to model how the mean depends on other variables → `regress`.

Don't switch silently — explain why the alternative is a better fit and confirm with the user before changing tools.

---

## Style notes

- **Pedagogical tone.** Many users will be students learning hypothesis testing for the first time. Always state the null and alternative hypotheses explicitly in plain English using the variable's real-world units. Always tie significance back to a 5% threshold (or whatever the user specified). Always interpret `diff` as a substantive effect size, not just a p-value.
- **Use the units from the description file.** Generic "the mean is 1953" is much weaker than "average monthly demand per store is 1953 units". The sidecar `.md` exists for exactly this reason.
- **Pick `alt_hyp` from the business question, not from the data.** Looking at the sample, seeing it's above `comp_value`, and *then* picking `alt_hyp="greater"` is p-hacking. The direction should be set by the decision the test is supposed to inform.
- **Don't over-engineer.** This is an interactive, exploratory workflow, not a production pipeline. Don't wrap things in functions, don't add try/except scaffolding, don't write `if __name__ == "__main__"`. Plain top-level code that the student can copy line by line.
- **One step at a time.** After Step 4 (run) wait for the user to look at the summary before barreling into plots or probability calculations. After Step 5 (interpretation) ask whether they want the diagnostic plot before running it.

## Growing this skill

This skill is one of a family of `pyrsm-basics-*` skills covering the `pyrsm.basics` module. When `single_mean` isn't the right tool, point the user to the sibling skill named for the analysis they need (`pyrsm-single-prop`, `pyrsm-compare-means`, `pyrsm-goodness`, `pyrsm-prob-calc`, etc.). The workflow shape — ask for absolute path → load via `scripts/load_data.py` → read sidecar → propose spec → run → interpret — is shared across the family, so users moving between skills should find a familiar rhythm.
