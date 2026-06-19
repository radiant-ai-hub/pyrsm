---
name: pyrsm-combine
description: Combine two polars DataFrames in Python using the pyrsm library's `combine` function — supporting inner / left / right / full / semi / anti joins, vertical and horizontal binds, and set operations (intersect / union / setdiff). Use this skill whenever a student or analyst wants to merge two tables on a key, stack two tables of the same schema, identify rows that appear in both tables (or in one but not the other), or work with the pyrsm package for any two-DataFrame combining task. The output is a polars DataFrame the user can keep chaining onto. Triggers include phrases like "join these two tables", "merge on customer_id", "left join", "stack two datasets", "concatenate rows", "find rows in A but not in B", "anti join to find unmatched", "set intersection of two tables", "bind columns side by side", or any mention of combining two tabular datasets in a marketing/business analytics context.
---

# pyrsm combine workflow

This skill walks the user through combining two polars DataFrames using `pyrsm.eda.combine` — from loading two datasets to choosing the right join / bind / set operation and producing the combined result. It is designed for students learning data combination in a business/marketing analytics context.

For deep reference on the `combine` API, all 11 supported `how` values, automatic dtype alignment, the `add=` parameter for selecting columns from `y`, and worked examples, see `references/combine.md`.

## Output policy

Two rules apply to **every** pyrsm call you make in this skill:

1. **Show the full output verbatim.** When you call `.summary()`, `.predict()`, or any pyrsm method that prints results, paste the **complete printed output** into your response — every line of the coefficient table, every statistic, every row. Do NOT abbreviate. Do NOT replace it with a high-level "main takeaways" bullet list. Do NOT skip the table to "save space". The user is reading your response because they want to see the actual numbers, not your summary of them.

2. **Provide the full pedagogical interpretation, not lame bullets.** When walking through results, use the detailed templates in `references/<name>.md`: state hypotheses explicitly, walk every coefficient or cell with its real-world units from the sidecar description, give the significance verdict at the 5% level, use the "holding constant" qualifier (where applicable), and tie back to the business question. A response like "X is positive and significant; Y is positive and significant; Z is not" is exactly the failure mode this skill is built to prevent.

3. **Lead with the requested output.** When the user asks for a specific plot, table, or diagnostic (e.g., "create and interpret PDP", "show me the residual dashboard", "generate PIP", "give me the confusion matrix", "plot the gains chart"), your interpretation must focus on THAT output — not on a re-interpretation of the model's coefficient table or summary. Pi's failure pattern is to say *"I'll generate X and then interpret the fitted results"* and then quietly produce a coefficient walk-through. The user already saw the coefficient interpretation in the previous turn; do not repeat it unless they explicitly ask. Lead with the requested output's content, using the dedicated template for that output if the skill has one (Step 9 for PIP, Step 10 for PDP, etc.).

If your context is genuinely constrained and you cannot reproduce the full output, say so explicitly — never silently abbreviate.


---

## Step 1 — Ask for the data locations (two files)

The very first thing to do is identify *both* datasets the user wants to combine. They almost always need to be in two files (or already loaded as polars DataFrames from a previous step). Require an **absolute path** for each.

Example phrasing:

> Before we start, I need the **absolute paths** to both data files you want to combine. For example: `/Users/yourname/Downloads/superheroes.parquet` and `/Users/yourname/Downloads/publishers.parquet`. Which one should be the "left" table and which the "right" one?

If either path is relative or has a `~`, ask for the absolute path. If either file doesn't exist, say so plainly.

## Step 2 — Load both DataFrames

Use `scripts/load_data.py` on each file:

```bash
python scripts/load_data.py "<abs-path-to-x>"
python scripts/load_data.py "<abs-path-to-y>"
```

The script prints a JSON report for each. **Pay special attention to:**

- **The column lists.** For a join, both DataFrames must share at least one column (or have differently-named columns linked via `left_on` / `right_on`).
- **The dtypes of the would-be join keys.** Mismatches (one is `Categorical`, the other is `String`; one is `Int`, the other is `Float`) are common. `combine` auto-aligns dtypes (see Step 6) but it's worth being aware of.

### Sidecar description files

If sidecars exist, **read both before deciding what to do**. The descriptions tell you what each column means, which one is the natural join key, and whether the tables are intended to be merged or stacked.

## Step 3 — Decide what kind of combine

Confirm the combine specification with the user. The `how` parameter has 11 values across three families:

### Joins (need a key)

| `how` | Keeps from `x` | Keeps from `y` | Adds `y`'s non-key columns? | When to use |
| --- | --- | --- | --- | --- |
| `"inner"` | matched only | matched only | yes | "Only complete matches." |
| `"left"` | all | matched only | yes (nulls when no match) | "Keep all of x; add y where possible." |
| `"right"` | matched only | all | yes (nulls when no x match) | "Keep all of y; add x where possible." |
| `"full"` | all | all | yes (nulls on either side) | "Keep everything from both." |
| `"semi"` | matched only | none (filter only) | no | "Filter x to rows that have a y match." |
| `"anti"` | unmatched only | none (filter only) | no | "Find rows in x with NO match in y." |

### Binds (no key)

| `how` | What it does | Requires |
| --- | --- | --- |
| `"bind_rows"` | Vertical concat (stack rows) | Same or overlapping columns; missing columns become `null` |
| `"bind_cols"` | Horizontal concat (side by side) | Same number of rows; no overlapping column names |

### Set operations (rows compared on all columns)

| `how` | What it does |
| --- | --- |
| `"intersect"` | Rows present in BOTH datasets |
| `"union"` | All rows from both, deduplicated |
| `"setdiff"` | Rows in `x` but NOT in `y` |

### Decision flow

> Are both datasets the **same schema** (intended to be stacked / set-operated on)? → `bind_rows`, `union`, `intersect`, `setdiff`.
> Are they **different schemas linked by a key**? → one of the joins.
> Do they have the **same number of rows** with NO shared key but you want them side-by-side? → `bind_cols` (rare, use with caution).

Confirm the spec, including which is `x` (left), which is `y` (right), and which `on` key to use.

Do a quick sanity check before running:

- **Join key existence**: the `on` column must exist in both tables. Or, if names differ, use `left_on` and `right_on`.
- **Key cardinality**: if either side has duplicate keys, the join can multiply rows. Run `df[key].n_unique()` vs `df.height` to check.
- **Type mismatch**: `combine` will auto-align (Cat ↔ Str → Cat, Int ↔ Float → Float, fallback to String), but warn the user if a non-obvious cast is about to happen.

## Step 4 — Run combine

```python
import polars as pl
import pyrsm as rsm

# 1. Load both data sets (absolute paths)
x = pl.read_parquet("<abs-path-to-x>")
y = pl.read_parquet("<abs-path-to-y>")

# 2. Combine
result = rsm.eda.combine(
    x, y,
    on="<key>",                   # or use left_on/right_on for different names
    how="left",                   # or any of the 11 options
    add=None,                     # optional: list of cols from y to include
    suffix="_right",              # for overlapping non-key columns
)

# 3. Inspect
print(result)
print(f"Input shapes: x = {x.shape}, y = {y.shape}; output = {result.shape}")
```

Notes:
- `combine` accepts polars DataFrames and LazyFrames.
- `add` is useful when `y` has many columns and you only want a few from it (e.g., looking up a single field).
- `suffix` controls the suffix on non-key columns that have the same name in both.

## Step 5 — Interpret the output

Walk the user through the result:

1. **Describe the shape transformation.** "Input: x had `<N_x>` rows × `<C_x>` cols; y had `<N_y>` rows × `<C_y>` cols. Output: `<N>` rows × `<C>` cols using `how=<how>`."

2. **Check for unexpected row counts.** This is the single most important diagnostic for joins:
   - **`how="inner"`**: result rows = rows in x with a key match in y (multiplied by duplicate keys in y).
   - **`how="left"`**: result rows = rows in x (each unmatched x-row appears once with nulls; each x-row with k matches in y appears k times).
   - **`how="full"`**: result rows = rows in x with matches + rows in x without matches + rows in y without matches.
   - **Unexpectedly large result**: likely caused by duplicate keys on the y side (many-to-one became many-to-many).
   - **Unexpectedly small result**: likely caused by `inner` join silently dropping rows (use `left` to see what's missing, or `anti` to inspect the unmatched).

3. **Look at the nulls (for outer joins).** A `left` join populates `null` in y-columns for rows of x that didn't match. Always check: how many nulls did we introduce, and are those the rows we expected to be unmatched?

4. **Translate to a business conclusion.** Don't stop at "the join worked". Tie the output back to the question. For the superheroes example: "After left-joining publishers onto superheroes by `publisher`, we have 7 rows (matching the 7 superheroes). Hellboy is the only superhero whose publisher ('Dark Horse Comics') is not in the publishers table, so his `yr_founded` is null. The other 6 superheroes have valid founding years."

### Don't reach for `inner` reflexively — and always sanity-check row counts

The classic combine failure modes:

- **Silently dropping rows with `inner`**. The default is `inner`, which is the safest only when you trust the keys to be complete. Otherwise it silently throws away unmatched x-rows. If in doubt, start with `left` (which preserves x) and inspect the nulls.
- **Multiplying rows when the y-key is duplicated**. A "lookup table" should have one row per key — duplicates produce a cross product. Always check `y[key].n_unique() == y.height` before joining.
- **Misordering `x` and `y` and getting the wrong join direction.** `left` keeps x, `right` keeps y. Swapping the tables flips the meaning of `left` and `right`.
- **Using `bind_cols` when there's no logical row alignment.** Concatenating columns side-by-side is only meaningful if row i of x corresponds to row i of y. Without a shared logical ordering, it's a near-guaranteed bug.

So: **always state the expected row count before running**, and verify the actual result matches. Bring out `anti` to inspect unmatched rows when needed.

## Step 6 — Automatic dtype alignment

The `combine` function automatically aligns join-key dtypes between `x` and `y`, so you don't usually need to cast manually. The rules:

- **Enum vs Other** (Enum vs String / Categorical) → cast the Other side to Enum (preserves the categorical levels).
- **Categorical vs String** → cast the String side to Categorical.
- **Int vs Float** → cast the Int side to Float (lossless upcast).
- **Anything else** → cast both sides to String (safe fallback).

This is convenient but worth knowing about: if you join on a column that's `Int` in one table and `String` in the other, both end up as `String` in the output. That can cascade into other operations.

If you want to avoid auto-casting, pre-cast the columns yourself before calling `combine` — polars will use whatever types you give it.

## Step 7 — Extend with polars

`rsm.eda.combine` returns a `pl.DataFrame`. Common next steps:

### Filter for unmatched

```python
combined = rsm.eda.combine(x, y, on="customer_id", how="left")
unmatched = combined.filter(pl.col("customer_email").is_null())   # rows where y had no match
```

### Compute the join-driven shift in row count

```python
print(f"Inner kept {inner.height} of x's {x.height} rows ({inner.height / x.height * 100:.1f}%)")
```

### Find rows present in both, via `intersect`

```python
both = rsm.eda.combine(x, y, how="intersect")   # rows present in both, on all columns
```

### Chain a transform

```python
result = (
    rsm.eda.combine(x, y, on="id", how="left")
    .with_columns(merged=pl.col("col_x") + pl.col("col_y").fill_null(0))
    .filter(pl.col("status") != "deleted")
)
```

### Combine three or more tables

`combine` is two-table only. Chain multiple calls:

```python
a_b   = rsm.eda.combine(a, b, on="key", how="left")
a_b_c = rsm.eda.combine(a_b, c, on="key", how="left")
```

## Step 8 — Related EDA tools (when to switch)

`combine` is the right tool when you have two DataFrames to merge or stack. If the situation is different, suggest:

- **Single-DataFrame reshaping** → `pyrsm-pivot` (wide), `pyrsm-unpivot` (long).
- **Per-column summary stats on one DataFrame** → `pyrsm-explore`.
- **Plot the combined result** → `pyrsm-visualize`.
- **Distribution analysis on a single DataFrame** → `pyrsm-distr`.

Don't switch silently — explain why the alternative is a better fit and confirm with the user before changing tools.

---

## Style notes

- **Pedagogical tone.** Always state which join type, why, and what the expected row count is. Always verify it after running.
- **Use the units from the description files.** "Joined customers and orders" is much weaker than "for each of the 1,200 customers, attached their most recent order date and amount where available".
- **Default to `left` when in doubt.** `left` preserves all of x's rows and exposes the unmatched ones via nulls — it's the most informative join for diagnostic work. Move to `inner` only after confirming you don't need the unmatched rows.
- **Check key cardinality before joining.** A duplicated key on the right side multiplies rows. Always run `y[key].n_unique() == y.height` (or accept the implications).
- **`anti` is your friend for diagnostics.** "Which rows of x didn't find a match?" = `combine(x, y, how="anti")`.
- **For set operations, all columns must match.** `intersect`, `union`, `setdiff` compare entire rows, not just keys.
- **Don't over-engineer.** Plain top-level code that the student can copy line by line.
- **One step at a time.** After Step 4 wait for the user to verify the row count before barreling into further transformations.

## Growing this skill

This skill is one of a family of `pyrsm-eda-*` skills covering the `pyrsm.eda` module. When `combine` isn't the right tool, point the user to the sibling skill named for the operation they need (`pyrsm-pivot`, `pyrsm-unpivot`, `pyrsm-explore`, `pyrsm-distr`, `pyrsm-visualize`). The workflow shape — ask for absolute path → load via `scripts/load_data.py` → read sidecar → propose spec → run → interpret → extend with polars — is shared across the family.
