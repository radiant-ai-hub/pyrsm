---
name: pyrsm-single-prop
description: Run and interpret one-sample proportion tests in Python using the pyrsm library's `single_prop` class. Use this skill whenever a student or analyst wants to compare a sample proportion (a yes/no, success/failure, or two-level categorical outcome) to a hypothesized population proportion, choose between an exact binomial test and a z-test approximation, build or read a confidence interval for a proportion, decide whether brand preference / churn rate / response rate / defect rate differs from a benchmark, or work with the pyrsm package for any one-sample-proportion task — even if they don't explicitly say "pyrsm" or "single_prop". Triggers include phrases like "is the brand preference really 10%", "test whether response rate exceeds X", "one-sample binomial test", "build a 95% CI for a proportion", "compare a yes/no outcome to a benchmark", "what fraction of customers said yes vs the target", or any mention of testing a binary outcome's prevalence against a target value in a marketing/business analytics class.
---

# pyrsm single-proportion workflow

This skill walks the user through a complete one-sample-proportion hypothesis test using the `pyrsm.basics.single_prop` class — from loading their data to translating the test output into a plain-English decision their manager would understand. It is designed for students learning hypothesis testing in a business/marketing analytics context, so default to clear, pedagogical explanations rather than terse statistical output.

For deep reference on the `single_prop` API, binomial-exact vs z-test selection, Wilson confidence intervals, and the three equivalent ways to evaluate the test (p-value, confidence interval, critical number-of-successes), see `references/single-prop.md`. Read it whenever the user asks a question that goes beyond the basic fit-and-summary loop.

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

> Before we start, can you give me the **absolute path** to the data file you want to analyze? For example: `/Users/yourname/Downloads/consider.parquet`. Any of `.parquet`, `.csv`, `.tsv`, `.feather`, `.arrow`, `.xlsx`, or `.json` will work.

If the user gives you a relative path, a `~`-style path, or just a filename, ask again for an absolute path rather than guessing. If the path doesn't exist, say so plainly and ask them to double-check.

## Step 2 — Load the data into a polars DataFrame

Use `scripts/load_data.py` to do the loading. It handles file-type detection by extension, reads the file with the appropriate polars reader, and also looks for a sidecar markdown description file in the same folder.

Run it like this:

```bash
python scripts/load_data.py "<absolute-path-to-data>"
```

The script prints a small JSON report describing the loaded dataset (path, shape, column names with dtypes) and the path of any sidecar `.md` it found. Use that to plan your next step. After this initial probe, in the actual analysis code, you will do the equivalent inside the Python session you'll run the test in — see Step 4.

### Sidecar description file

In pyrsm-style course materials, every dataset is shipped alongside a markdown file that documents what the variables mean. The convention is `<dataset>_description.md` next to `<dataset>.parquet` (e.g., `consider.parquet` ↔ `consider_description.md`), but other "very similar" filenames are also common in the wild. The loader script searches for these patterns automatically.

If a sidecar is found, **read it before proposing what to test**. The description tells you what the categorical variable measures, what its levels are, and which level is the substantive "success" for the question at hand. This is the difference between a generic "the proportion of yes is significantly different from 10%" and a useful "9.3% of sampled consumers say they'd consider this brand of car — the data are consistent with the company's 10% threshold and do not justify the additional advertising spend."

If no sidecar is found, say so and ask the user to briefly describe the variable. Don't invent meanings.

## Step 3 — Decide together what to test

Once the data and (optionally) the description are in hand, propose a test specification and confirm it with the user before running. You need six things:

- **Variable (`var`)** — one column. Must contain a categorical (string / Enum / Categorical) variable with at least the level you want to test.
- **Level (`lev`)** — the level of `var` that counts as "success" for this test. This is a **substantive choice** tied to the research question, not a default. Ask: "which level is the one we're measuring the rate of?" For a `yes`/`no` column where the question is "what fraction would consider the brand?", `lev="yes"`.
- **Comparison value (`comp_value`)** — the population proportion under H₀. Must be strictly between 0 and 1 (the class raises if you pass 0 or 1). It is a substantive choice: a target, benchmark, contract minimum, or prior published rate. **Ask where the comparison value comes from** if the user doesn't volunteer it.
- **Alternative hypothesis (`alt_hyp`)** — `"two-sided"`, `"greater"`, or `"less"`. Pick *before* looking at the data, based on the decision being made (same logic as in `pyrsm-single-mean`).
- **Confidence level (`conf`)** — almost always `0.95`.
- **Test type (`test_type`)** — `"binomial"` (default, exact binomial test) or `"z"` (normal-approximation z-test with Wilson CI). See Step 6 — the choice matters.

If the description file or the user's question makes the choice obvious (e.g., "is brand preference below 10%?"), state your proposed specification — `var="consider"`, `lev="yes"`, `comp_value=0.1`, `alt_hyp="less"`, `conf=0.95`, `test_type="binomial"` — and ask the user to confirm.

Do a quick sanity check before running:

- Is `var` truly two-level (or are you collapsing a multi-level categorical down to "is this level vs anything else")? If three or more levels matter as distinct outcomes, this is the wrong tool — consider `goodness` instead.
- Is `lev` actually a level of `var`? A typo (`"Yes"` vs `"yes"`) silently yields `ns = 0`, which makes the test return nonsense. Check `df[var].value_counts()` if unsure.
- Does `var` have many missing values? The class drops them automatically (showing them as `n_missing`), but flag it so the user knows the effective sample size.

## Step 4 — Run the test

Write a short, runnable Python script (or run it interactively if you have a Python REPL available). Keep it minimal and readable — students will look at this code:

```python
import polars as pl
import pyrsm as rsm

# 1. Load the data (absolute path)
data_path = "<absolute-path>"
df = pl.read_parquet(data_path)   # adjust reader for the actual file type

# 2. Run the one-sample proportion test
sp = rsm.basics.single_prop(
    data={"<dataset_name>": df},        # dict so the dataset name shows up in the summary
    var="<column>",
    lev="<success-level>",
    alt_hyp="<two-sided|greater|less>",
    conf=0.95,
    comp_value=<benchmark>,
    test_type="binomial",               # or "z"
)

# 3. Inspect
sp.summary()
```

Notes:
- Pass the data as `{"name": df}` (a dict with one key) rather than just `df`, so the summary header prints a meaningful dataset name instead of `"Not provided"`.
- `pyrsm` accepts polars DataFrames directly — no need to convert to pandas first.
- The summary prints two tables: descriptive statistics (p, ns, n, n_missing, sd, se, me) on top, and the hypothesis-test row on the bottom. For a binomial test the test row shows `diff`, `ns` (number of successes), `p.value`, and CI bounds. For a z-test it shows `diff`, `z.value`, `p.value`, and CI bounds.

## Step 5 — Interpret the output

This is where most of the value is. Walk the user through the summary in roughly this order — see `references/single-prop.md` for the detailed templates and worked examples:

1. **State the hypotheses in words.** Read them straight off the summary header:
   - H₀: the proportion of `<lev>` in `<var>` is equal to `<comp_value>`.
   - Hₐ: the proportion of `<lev>` in `<var>` is `<greater than | less than | not equal to>` `<comp_value>`.
   Repeat them in the user's vocabulary using the units from the sidecar.

2. **Describe the sample.** One sentence: "the sample contains `<n>` observations (`<n_missing>` missing), of which `<ns>` are `<lev>` — a sample proportion of `<p>`, with standard error `<se>`."

3. **State the test result three equivalent ways.** This is the pedagogical core of a single-proportion test:
   - **p-value**: "p = `<p.value>`, which is `<smaller | not smaller>` than 0.05, so we `<reject | fail to reject>` H₀."
   - **Confidence interval**: "the 95% CI for the population proportion is [`<lo>`, `<hi>`]. Because `<comp_value>` `<is | is not>` inside this interval, we `<reject | fail to reject>` H₀." (For a one-sided test the interval is half-open: `[0, hi]` for `"less"`, `[lo, 1]` for `"greater"`.)
   - **Critical number-of-successes** (binomial test only): "we observed `<ns>` successes out of `<n>` trials. Under H₀ (Binomial(n=`<n>`, p=`<comp_value>`)) the critical number of successes for the rejection region is approximately `<crit>` (computed via `prob_calc('binom', n=<n>, p=<comp_value>, plb=alpha)` for a `"less"` test, or `pub=1-alpha` for a `"greater"` test). Because `<ns>` `<is | is not>` in the rejection region, we `<reject | fail to reject>` H₀."

4. **Translate to a business conclusion.** Don't stop at "we reject H₀". Tie it back to the original question, with units. For the consider example: "the sample proportion of consumers who would consider this brand is 9.3% — we cannot rule out that the true population rate is at or above the 10% threshold, so the data do not justify additional advertising spend."

5. **Effect size.** Always report **how far** the sample proportion is from `comp_value`, not just whether the difference is significant. The `diff` row in the summary is exactly this — usually 1-3 percentage points in marketing surveys. With n in the thousands, even a tiny `diff` can produce a small p-value; whether the magnitude matters for the decision is a separate question.

### Don't read significance off the p-value alone

After looking at the summary, students very often jump straight from "p < 0.05" to "the result is real and meaningful" — or, in the other direction, from "p = 0.06" to "the rate isn't different from the benchmark". Both moves are wrong, and the skill should push back on them.

The reasons, stated for the student:

- **A p-value just under or just over 0.05 is not a hard cliff.** p = 0.04 and p = 0.06 are nearly identical pieces of evidence. The 0.05 threshold is a convention, not a physical law.
- **A significant p-value is not the same as a large practical effect.** With n in the tens of thousands, a 0.1-percentage-point difference can land at p < .001. Report `diff` in percentage points and consider whether that magnitude would change the business decision.
- **The confidence interval contains strictly more information than the p-value.** It tells you the range of population proportions that are plausibly consistent with the data. Lead with the CI when explaining the result.

So: **when interpreting, do not lean on the p-value alone**. Walk all three views, discuss the magnitude of `diff` in percentage points, and only then state the conclusion.

## Step 6 — Choose between binomial-exact and z-test

`single_prop` supports two test variants via `test_type`:

- **`test_type="binomial"`** (default) — exact test using the binomial distribution. Wraps `scipy.stats.binomtest`. The CI is the Clopper–Pearson exact interval. **Always correct**, but the p-value can be slightly conservative.
- **`test_type="z"`** — z-test using the normal approximation to the binomial. The CI is the Wilson score interval (the same one R's `prop.test` uses). Faster, gives a `z.value` for direct comparison to a critical-z, but **only reliable when both `n * comp_value` and `n * (1 - comp_value)` are at least ~5–10** (a CLT-style sample-size requirement on the binomial).

### When to use which

- **n × p₀ and n × (1−p₀) both ≥ ~10** → either is fine. The z-test gives a nicer continuous test statistic; the binomial-exact is more conservative but more exact. For a class assignment, default to the binomial.
- **n × p₀ < ~5 or n × (1−p₀) < ~5** → use the binomial-exact. The CLT approximation underlying the z-test breaks down, and the z-test's p-value and CI become unreliable.
- **The user explicitly asks for a z-test** (e.g., because they want a z-statistic to compare to a critical value, or because their textbook teaches it that way) → fine, but check the sample-size requirement first.

### Why this matters

This is the most common pitfall in single-proportion tests as students learn them: applying the z-test to a small sample with a rare outcome, getting a tidy-looking output, and trusting it. The z-test will happily return a p-value and a Wilson CI for n=20, p₀=0.05 — but neither number means what the student thinks it means, because the underlying CLT approximation hasn't kicked in.

So: **always check `n * comp_value ≥ 5` and `n * (1 - comp_value) ≥ 5` before using `test_type="z"`**. If either fails, run the binomial-exact instead (or in addition) and report results from that. If the user insists on the z-test for a small sample, explain the assumption violation and show both tests' outputs side by side — the discrepancy is the lesson.

## Step 7 — Plot (offer, don't force)

After the basic interpretation, **offer** to run the bar plot.

```python
sp.plot()
```

The default plot is a simple bar of the proportion of each level in `var`. It is useful for sanity-checking the sample composition; it does not visualize the test itself. There is no `"hist"` overlay-with-CI plot like there is for `single_mean`.

## Step 8 — Probability-calculator companion (optional)

The pyrsm probability calculator can produce the **critical number of successes** for the binomial-exact test, or the **critical z** for the z-test:

```python
# Binomial: critical ns at alpha = 0.05 for a one-sided "less" test, n = 1000, p_0 = 0.10
pc = rsm.basics.prob_calc("binom", n=1000, p=0.1, plb=0.05)
pc.summary()

# Binomial: p-value as P(X <= observed) under H0
pc = rsm.basics.prob_calc("binom", n=1000, p=0.1, lb=93)
pc.summary()

# Normal: critical z at alpha = 0.05 for a one-sided "less" test
pc = rsm.basics.prob_calc("norm", mean=0, sd=1, plb=0.05)
pc.summary()
```

For the consider example, `prob_calc("binom", n=1000, p=0.1, plb=0.05)` returns a critical number of 85 — meaning under H₀ we'd expect 85 or fewer successes only 5% of the time. We observed 93, which is above 85, so we cannot reject. The p-value calculation (`lb=93`) returns P(X ≤ 93) ≈ 0.249, matching the test's `p.value`.

If the user wants more on `prob_calc`, route them to the `pyrsm-prob-calc` skill.

## Step 9 — Related tests (when to switch)

`single_prop` is the right tool when the response is a single two-level categorical variable and you have one group. If the situation is different, suggest:

- Response is a continuous numeric variable → `single_mean`.
- Comparing the proportion of one level across two or more groups → `compare_props`.
- Three or more levels at once, testing whether the distribution matches an expected one → `goodness`.
- Two categorical variables, testing for association → `cross_tabs`.
- Modeling how the probability of `lev` depends on other variables → `logistic`.

Don't switch silently — explain why the alternative is a better fit and confirm with the user before changing tools.

---

## Style notes

- **Pedagogical tone.** Many users will be students learning hypothesis testing for the first time. Always state the null and alternative hypotheses explicitly in plain English using the variable's real-world units. Always tie significance back to a 5% threshold (or whatever the user specified). Always interpret `diff` as a substantive effect size in percentage points.
- **Use the units from the description file.** Generic "the proportion is 0.093" is much weaker than "9.3% of sampled consumers say they'd consider the brand". The sidecar `.md` exists for exactly this reason.
- **Pick `lev`, `alt_hyp`, and `comp_value` from the business question, not from the data.** Looking at the sample and *then* choosing one-sided or *then* picking a comp_value that lands just above/below the rate is a form of p-hacking.
- **Check the z-test sample-size requirement.** This is the unique critical concept for single-proportion tests. Always look at `n × comp_value` and `n × (1 − comp_value)` before defaulting to `test_type="z"`.
- **Don't over-engineer.** Plain top-level code that the student can copy line by line.
- **One step at a time.** After Step 4 wait for the user to look at the summary before barreling into plots or probability calculations.

## Growing this skill

This skill is one of a family of `pyrsm-basics-*` skills covering the `pyrsm.basics` module. When `single_prop` isn't the right tool, point the user to the sibling skill named for the analysis they need (`pyrsm-single-mean`, `pyrsm-compare-props`, `pyrsm-goodness`, `pyrsm-cross-tabs`, `pyrsm-prob-calc`, etc.). The workflow shape — ask for absolute path → load via `scripts/load_data.py` → read sidecar → propose spec → run → interpret — is shared across the family.
