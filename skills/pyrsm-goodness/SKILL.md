---
name: pyrsm-goodness
description: Run and interpret chi-squared goodness-of-fit tests in Python using the pyrsm library's `goodness` class. Use this skill whenever a student or analyst wants to compare the observed distribution of a single categorical variable to an expected distribution (uniform, census-based, historical, or theoretical), check which cells drive any deviation via standardized residuals, or work with the pyrsm package for any one-variable chi-squared task — even if they don't explicitly say "pyrsm" or "goodness". Triggers include phrases like "test if this categorical variable follows expected proportions", "chi-square goodness-of-fit", "does the income split match the census 70/30", "is the sample evenly distributed across categories", "are the levels of X consistent with a hypothesized distribution", "test if dice are fair", "compare observed vs expected frequencies", or any mention of testing a single discrete distribution against a benchmark in a marketing/business analytics class.
---

# pyrsm goodness-of-fit workflow

This skill walks the user through a chi-squared goodness-of-fit test on a single categorical variable using the `pyrsm.basics.goodness` class — from loading their data to translating the test output into a plain-English decision and (when the test rejects) identifying which cells drive the deviation. It is designed for students learning hypothesis testing in a business/marketing analytics context.

For deep reference on the `goodness` API, choice of expected probabilities, standardized-deviation interpretation, sample-size effects on chi-squared tests, and the relationship to `cross_tabs`, see `references/goodness.md`.

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

> Before we start, can you give me the **absolute path** to the data file you want to analyze? For example: `/Users/yourname/Downloads/newspaper.parquet`. Any of `.parquet`, `.csv`, `.tsv`, `.feather`, `.arrow`, `.xlsx`, or `.json` will work.

If the user gives you a relative path, a `~`-style path, or just a filename, ask again for an absolute path rather than guessing. If the path doesn't exist, say so plainly and ask them to double-check.

## Step 2 — Load the data into a polars DataFrame

Use `scripts/load_data.py` to do the loading. It handles file-type detection by extension, reads the file with the appropriate polars reader, and also looks for a sidecar markdown description file in the same folder.

Run it like this:

```bash
python scripts/load_data.py "<absolute-path-to-data>"
```

The script prints a small JSON report describing the loaded dataset (path, shape, column names with dtypes) and the path of any sidecar `.md` it found. After this initial probe, in the actual analysis code, you will do the equivalent inside the Python session you'll run the test in — see Step 4.

### Sidecar description file

In pyrsm-style course materials, every dataset is shipped alongside a markdown file that documents what the variables mean. The convention is `<dataset>_description.md` next to `<dataset>.parquet` (e.g., `newspaper.parquet` ↔ `newspaper_description.md`). The loader script searches for this and several similar patterns automatically.

If a sidecar is found, **read it before proposing what to test**. The description tells you what the categorical variable measures, what its levels are, and — critically — whether there is a substantive benchmark for the *expected* distribution. The expected probabilities should come from this benchmark whenever possible (e.g., census proportions, historical baseline, contract minimum), not from a default uniform assumption.

If no sidecar is found, say so and ask the user to briefly describe the variable. Don't invent meanings.

## Step 3 — Decide together what to test

Once the data and (optionally) the description are in hand, propose a test specification and confirm it with the user before running. You need three things:

- **Variable (`var`)** — one column. Must be categorical (string / Enum / Categorical) with at least two levels.
- **Expected probabilities (`probs`)** — a tuple of floats, one per level, **summing to ~1.0** (the class checks for sum ∈ [0.999, 1.001] and raises otherwise). The order of `probs` corresponds to **alphabetically sorted levels** of `var`, not the order they appear in the data. If `probs=None`, the class defaults to a **uniform distribution** (`1/k` for each of the `k` levels) — but **uniform is a default, not a substantive choice**. See Step 6.
- **(Implicit) Significance level** — the test is fixed-form chi-squared with `k − 1` degrees of freedom. The conventional α = 0.05 applies. Confidence-level configuration is not exposed by `goodness` (unlike single_mean / single_prop).

If the description file or the user's question makes the choice obvious (e.g., "is the income split 50-50?"), state your proposed specification — `var="Income"`, `probs=(0.5, 0.5)` for the uniform case — and ask the user to confirm. **Critically, ask "where does the comparison distribution come from?"** before defaulting to uniform.

Do a quick sanity check before running:

- How many levels does `var` have? `df[var].value_counts()` or `df.group_by(var).len().sort(var)` tells you. The class will sort them alphabetically before applying `probs`.
- Do `probs` sum to ~1? If not, the class raises. If the user gave you percentages (e.g., 30, 70), divide by 100 first.
- Are any expected counts likely to be **below 5**? Multiply `len(df) * min(probs)`. If the result is < 5, the chi-squared p-value is biased — the test is unreliable until you collapse categories. See Step 7.

## Step 4 — Run the test

Write a short, runnable Python script (or run it interactively if you have a Python REPL available). Keep it minimal and readable — students will look at this code:

```python
import polars as pl
import pyrsm as rsm

# 1. Load the data (absolute path)
data_path = "<absolute-path>"
df = pl.read_parquet(data_path)   # adjust reader for the actual file type

# 2. Run the goodness-of-fit test
gf = rsm.basics.goodness(
    {"<dataset_name>": df},        # dict so the dataset name shows up in the summary
    var="<column>",
    probs=(<p1>, <p2>, ...),       # ordered by sorted level names; None → uniform
)

# 3. Inspect
gf.summary(output=["observed", "expected", "chisq", "dev_std"])
```

Notes:
- Pass the data as `{"name": df}` (a dict with one key) rather than just `df`, so the summary header prints a meaningful dataset name instead of `"Not provided"`.
- `pyrsm` accepts polars DataFrames directly.
- The `output` argument to `summary()` is a **list** controlling which tables print. Use all four (`"observed"`, `"expected"`, `"chisq"`, `"dev_std"`) on the first look — they are cheap and each tells the student something different. The chi-squared statistic, df, and p-value print in the footer regardless.

## Step 5 — Interpret the output

This is where most of the value is. Walk the user through the summary in roughly this order — see `references/goodness.md` for detailed templates and worked examples:

1. **State the hypotheses in words.** Read straight off the summary header:
   - H₀: the distribution of `<var>` is consistent with the specified probabilities `<probs>`.
   - Hₐ: the distribution of `<var>` is **not** consistent with the specified probabilities.
   Restate in the user's own vocabulary using the sidecar units (e.g., "H₀: high-income and low-income respondents are equally common in the population (50/50)").

2. **Walk the tables, in order:**
   - **Observed** — the actual counts. Highlight the totals.
   - **Expected** — `total × prob` for each level. Compare visually to observed.
   - **Chi-squared contributions** — `(observed − expected)² / expected` per cell. Their sum is the test statistic.
   - **Standardized deviations** — `(observed − expected) / sqrt(expected)`. Cells with `|dev_std| > 1.96` are individually deviating significantly from the expected proportion at α = 0.05 (rough rule of thumb).

3. **State the test result.** Read the footer: chi-squared statistic, df, p-value. State conclusion:
   - "Chi-squared = `<chi2>` on `<df>` df, p-value `<p>`. Because p `<is | is not>` smaller than 0.05, we `<reject | fail to reject>` H₀."

4. **If you reject, identify which cells drive it.** Use the `dev_std` table. Cells with `|dev_std| > 1.96` are over- or under-represented. State direction:
   - Positive `dev_std` → observed > expected for that cell.
   - Negative `dev_std` → observed < expected for that cell.

5. **Translate to a business conclusion.** Don't stop at "we reject H₀". Tie it back to the original question. For the newspaper Income example: "There are fewer high-income respondents (221) than the 50/50 null would predict (290) and correspondingly more low-income respondents (359). The deviation is highly significant (chi-sq 32.83, p < .001), with both cells contributing meaningfully (`|dev_std| ≈ 4.05` each, well above 1.96)."

### Don't read significance off the p-value alone — and don't default to uniform

The classic chi-squared goodness-of-fit failure modes:

- **Defaulting to uniform `probs` without thinking.** Uniform is rarely the right null hypothesis. For income, a US census 50/50 split is not the natural baseline — the underlying population may be 60/40 or 70/30. Pyrsm defaults to uniform when `probs=None`, but the **substantive null should come from the question**: census data, historical baseline, contract minimum, prior period's distribution. Always ask "where does the expected distribution come from?" before settling on uniform.
- **Large-n reject-on-anything.** With n in the thousands, chi-squared rejects any tiny deviation — a 1-percentage-point gap can come back as p < .001. A statistically significant result with a small effect size is not the same as a practically important one. Report `chi-sq / n` (a rough effect-size proxy) and standardized deviations alongside the p-value, and discuss whether the magnitude is decision-relevant.
- **Reading the omnibus chi-squared without the `dev_std` table.** The chi-squared statistic only says "the distribution is off somewhere"; it does not tell you *where*. Always look at standardized deviations to identify which cells drive the rejection.

So: **when interpreting, do not lean on the p-value alone**. Question the expected distribution, report effect-size context, and unpack the standardized deviations.

## Step 6 — Choosing the expected probabilities

This is the **unique pedagogical challenge** of the goodness test, and the place students most often go wrong.

### Uniform default — when is it actually right?

A uniform null (`probs=None`, i.e., `1/k` per level) is the right null only when:

1. The underlying random process should produce levels uniformly — e.g., testing whether a die is fair, whether random-assignment buckets are balanced, whether days-of-the-week have equal counts in a calendar-balanced sample.
2. There is no substantive prior expectation, and the test is purely exploratory ("does anything stand out?").

In all other cases, the right `probs` come from outside the data:

- **Census or population data.** "What proportion of the US population is high-income?" → use those proportions as `probs`.
- **Historical baseline.** "What was the share of each product in last year's sales?" → use last year's shares.
- **Contract or design.** "The contract says each operator gets 25% of the calls" → use `probs=(0.25, 0.25, 0.25, 0.25)`.
- **Theoretical model.** "Under Mendelian inheritance we expect 3:1 dominant:recessive" → use `probs=(0.75, 0.25)`.

### Order of `probs`

Pyrsm sorts the levels of `var` alphabetically before applying `probs`. So for a variable with levels `"High Income"` and `"Low Income"`, the first slot of `probs` applies to `"High Income"` (H < L), the second to `"Low Income"`. **Always verify the order before running**, especially when probs are asymmetric. Print the observed table once; the column order shows the sorted order.

### Probs must sum to 1

The class raises if `sum(probs)` is outside `[0.999, 1.001]`. If you have percentages, divide by 100. If you have raw counts as a prior, divide by their sum.

### When probs come from the same data — bias warning

If you derive `probs` from the same sample you are testing (e.g., fit a marginal distribution and then test against it), the test loses degrees of freedom and the p-value is biased toward fail-to-reject. The goodness test assumes the expected distribution is set *a priori*, not estimated from the test sample.

## Step 7 — Sample-size and expected-count assumptions

The chi-squared p-value is approximate. It relies on the test statistic following a chi-squared distribution under H₀, which is a large-sample asymptotic result.

### The expected-count rule

A conventional rule of thumb: **every expected cell count should be at least 5**. When some expected counts are smaller, the chi-squared p-value is biased.

- Check: `len(df) * min(probs)`. If < 5, the assumption is violated for at least one cell.
- Remedies:
  - **Collapse adjacent categories** to bring the smallest expected counts up. (E.g., for a 5-level rating "very poor", "poor", "neutral", "good", "very good" with very few "very poor" responses, collapse to "poor or worse" / "neutral" / "good or better".)
  - **Use an exact test.** Pyrsm does not expose an exact multinomial test directly; if collapsing is not acceptable, compute the exact p-value with `scipy.stats.multinomial` manually, or use a Fisher-style approach when only two levels are involved.
  - **Get more data.** Sometimes feasible, often not.

If the user proceeds with violated assumptions, **say so explicitly in the writeup**. Reporting p < .001 from a table with three expected counts below 5 is misleading.

### Large-n inflation

The opposite problem: very large samples reject any deviation, no matter how small. With n = 100,000 and a true distribution that is (0.5005, 0.4995) vs a hypothesized (0.5, 0.5), the test will reliably return p < .001 — but the deviation is irrelevant for any business decision. Always pair the p-value with a magnitude readout (the `dev_std` table is the easiest: how large are the standardized deviations relative to 1.96?).

## Step 8 — Plot (offer, don't force)

After the basic interpretation, **offer** to run the plots. The most useful ones:

```python
gf.plot(plots=["observed", "expected", "chisq", "dev_std"])
```

This composes four bar charts: observed counts, expected counts, chi-squared contributions, and standardized deviations. The `dev_std` plot includes dotted reference lines at ±1.96 — bars extending beyond those are individually significant deviations.

Use a single subset (`plots=["observed", "expected"]` or `plots=["dev_std"]`) when the user only wants one panel.

## Step 9 — Probability-calculator companion (optional)

The pyrsm probability calculator can produce the **critical chi-squared value** and **p-value** for any chi-squared statistic:

```python
# critical chi-squared at alpha = 0.05 with df = k - 1
pc = rsm.basics.prob_calc("chisq", df=<k-1>, pub=0.95)
pc.summary()

# p-value for an observed chi-squared statistic
pc = rsm.basics.prob_calc("chisq", df=<k-1>, ub=<observed_chi2>)
pc.summary()
```

For the newspaper Income example: critical chi-squared at α = 0.05 with df = 1 is 3.841. Observed 32.83 ≫ 3.841 → reject.

If the user wants more on `prob_calc`, route them to the `pyrsm-prob-calc` skill.

## Step 10 — Related tests (when to switch)

`goodness` is the right tool when you have one categorical variable and a hypothesized distribution. If the situation is different, suggest:

- Two-level categorical variable with a single benchmark proportion → `single_prop` (more direct API, supports binomial-exact).
- Two categorical variables, testing for association → `cross_tabs`.
- Single continuous variable, testing whether the mean matches a benchmark → `single_mean`.
- Categorical response that you want to *model* given covariates → `logistic` (binary) or downstream tools.

Don't switch silently — explain why the alternative is a better fit and confirm with the user before changing tools.

---

## Style notes

- **Pedagogical tone.** Many users will be students learning chi-squared tests for the first time. Always state the null and alternative hypotheses explicitly. Always tie significance back to a 5% threshold. Always interpret the chi-squared statistic alongside the standardized deviations.
- **Use the units from the description file.** Generic "the distribution is significantly different" is much weaker than "high-income readers are over-represented and low-income readers are under-represented relative to a 50/50 baseline".
- **Pick `probs` from the substantive question, not from the data.** Uniform default is convenient but rarely substantive. Always ask "where does the expected distribution come from?"
- **Check the expected-count assumption.** Before reporting p < .001 from any chi-squared test, multiply `n * min(probs)` and check it's ≥ 5. If not, say so.
- **Pair p-value with effect size.** With large n the chi-squared test rejects any deviation. The `dev_std` table is the natural effect-size view; standardized deviations of 0.5 are tiny even when the omnibus p is < .001.
- **Don't over-engineer.** Plain top-level code that the student can copy line by line.
- **One step at a time.** After Step 4 wait for the user to look at the tables before barreling into plots.

## Growing this skill

This skill is one of a family of `pyrsm-basics-*` skills covering the `pyrsm.basics` module. When `goodness` isn't the right tool, point the user to the sibling skill named for the analysis they need (`pyrsm-single-prop`, `pyrsm-cross-tabs`, `pyrsm-prob-calc`, etc.). The workflow shape — ask for absolute path → load via `scripts/load_data.py` → read sidecar → propose spec → run → interpret — is shared across the family.
