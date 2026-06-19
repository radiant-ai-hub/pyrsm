# pyrsm.eda.combine — reference

This file is the deeper reference for `pyrsm.eda.combine`. The main `SKILL.md` walks the workflow at a high level; come here for API details, all 11 `how` variants, automatic dtype alignment, and worked examples.

## Table of contents

1. Function signature
2. The three families: joins, binds, sets
3. The six join types in detail
4. The two bind types
5. The three set operations
6. Automatic join-key dtype alignment
7. The `add=` parameter — limiting columns from `y`
8. The `suffix` parameter — disambiguating overlap
9. Plain-English interpretation templates
10. Extending the output with polars
11. Worked examples (superheroes / publishers / avengers)
12. Common pitfalls

---

## 1. Function signature

```python
rsm.eda.combine(
    x,                            # pl.DataFrame or pl.LazyFrame (left/first)
    y,                            # pl.DataFrame or pl.LazyFrame (right/second)
    on=None,                      # str | list[str] | None — shared join key(s)
    how="inner",                  # one of 11 values, see below
    left_on=None,                 # str | list[str] | None — left-side keys (for renamed keys)
    right_on=None,                # str | list[str] | None — right-side keys
    add=None,                     # list[str] | None — limit which y columns are added
    suffix="_right",              # suffix for overlapping non-key columns
) -> pl.DataFrame
```

Returns a `pl.DataFrame`. Always.

`how` is validated against an internal `ALL_TYPES` set; unknown values raise `ValueError`.

For joins, you must provide either `on=` (same column name on both sides) or both `left_on=` and `right_on=` (different column names).

`add` is only used for joins (ignored for binds and set operations).

`suffix` defaults to `"_right"` (matching polars convention for join output).

## 2. The three families: joins, binds, sets

The 11 `how` values group into three families:

```python
JOIN_TYPES = {"inner", "left", "right", "full", "semi", "anti"}
BIND_TYPES = {"bind_rows", "bind_cols"}
SET_TYPES = {"intersect", "union", "setdiff"}
```

| Family | Schema requirement | Key requirement | What it does |
| --- | --- | --- | --- |
| Joins | Different OK | Yes | Combine columns based on key match |
| Binds | Compatible | No | Stack rows (`bind_rows`) or columns (`bind_cols`) |
| Sets | Identical | No | Row-level set operations (intersect / union / setdiff) |

## 3. The six join types in detail

### `how="inner"` (default)

Returns rows where the key exists in **both** `x` and `y`. Columns from both sides are included.

- Result rows ≤ min(rows in x with at least one match, rows in y with at least one match).
- With duplicate keys: a Cartesian product within each key group.

### `how="left"`

Returns **all** rows from `x`. For rows in x with a match in y, the matched y-columns are filled in. For unmatched x-rows, y-columns are `null`.

- Result rows = rows in x (or more if y has duplicate keys).
- The natural default for "augment x with information from y".

### `how="right"`

Returns **all** rows from `y`. For rows in y with a match in x, the matched x-columns are filled in. For unmatched y-rows, x-columns are `null`.

- Result rows = rows in y (or more if x has duplicate keys).
- Less common — usually swapping x and y and using `left` is clearer.

### `how="full"` (outer)

Returns rows from **either** side. Unmatched rows get `null` in the missing-side columns.

- Result rows = rows in x with match + rows in x without match + rows in y without match.
- Useful for finding all the things that exist in either dataset.

### `how="semi"`

Returns rows from `x` that have a match in `y`. **Only x's columns** are kept — y is used as a filter.

- Result rows ≤ rows in x.
- "Filter x to keys present in y."

### `how="anti"`

Returns rows from `x` that do **not** have a match in `y`. Only x's columns are kept.

- Result rows ≤ rows in x.
- "Find x-rows with no y match" — invaluable for diagnosing why an inner join lost rows.

## 4. The two bind types

### `how="bind_rows"`

Stacks rows of `x` and `y` vertically (concatenation). Uses polars' `diagonal` concat, which:

- Aligns columns by name.
- Fills missing columns with `null` (so partial overlap is allowed).

Result rows = rows in x + rows in y. Duplicates are kept (not deduplicated — see `union` for that).

Equivalent to SQL `UNION ALL`.

### `how="bind_cols"`

Concatenates `x` and `y` side-by-side (horizontal concat). Requires:

- Same number of rows in x and y.
- No overlapping column names (otherwise the polars horizontal concat raises).

Result rows = rows in x (= rows in y). Columns = cols in x + cols in y.

Use with extreme caution: the alignment is positional, not key-based. If row ordering of x and y doesn't have a meaningful correspondence, the result is nonsense.

## 5. The three set operations

All three operate on **entire rows** (matching on all columns, not just a key).

### `how="intersect"`

Rows that appear in **both** x and y. Implementation: `x.join(y, on=x.columns, how="semi")`.

- Result rows = rows of x present (as full row tuples) in y.
- Duplicates within x are preserved; duplicates within y are not. (More precisely: a row from x is kept if at least one matching row exists in y.)

### `how="union"`

All rows from both, **deduplicated**.

- Result rows = `pl.concat([x, y]).unique().height`.
- Equivalent to SQL `UNION` (which is `UNION ALL` + `DISTINCT`).

### `how="setdiff"`

Rows in `x` that are **not** in `y`. Implementation: `x.join(y, on=x.columns, how="anti")`.

- Result rows = rows of x not present (as full row tuples) in y.
- Equivalent to SQL `EXCEPT`.

## 6. Automatic join-key dtype alignment

A common annoyance with joins is dtype mismatch on the key. `combine`'s `_align_join_key_dtypes` handles this automatically with a priority order:

1. **Enum vs (String / Categorical)** → cast the other side to Enum (preserves the Enum's categories).
2. **Categorical vs String** → cast String side to Categorical.
3. **Int vs Float** → cast Int side to Float (lossless upcast).
4. **Fallback (e.g., Bool vs Int)** → cast both sides to String.

This means `combine(x_with_str_key, y_with_cat_key, on="key")` works without manual casting. The output may have a different dtype for the key than either input, however — keep that in mind when chaining.

### When to bypass auto-alignment

If you need the result's key column to be a specific dtype, pre-cast it explicitly before calling `combine`:

```python
x = x.with_columns(pl.col("key").cast(pl.String))
y = y.with_columns(pl.col("key").cast(pl.String))
rsm.eda.combine(x, y, on="key", how="left")
```

This is rarely needed for analysis purposes; mostly for downstream tools that require specific dtypes.

## 7. The `add=` parameter — limiting columns from `y`

By default a join brings in all of `y`'s columns. When `y` has many columns and you only want a few, pass `add=[...]`:

```python
result = rsm.eda.combine(
    x, y,
    on="customer_id",
    how="left",
    add=["email", "phone"],   # only these from y (plus the key)
)
```

Internally, `combine` selects `right_keys + add` from `y` before joining, so the resulting DataFrame has just those y-columns plus everything from x.

`add` is ignored for binds and set operations.

## 8. The `suffix` parameter — disambiguating overlap

When a non-key column has the same name in both `x` and `y`, polars (and `combine`) add a suffix to the y-side column to disambiguate.

Default: `suffix="_right"`. So `x` has `revenue` and `y` has `revenue` → output has `revenue` (from x) and `revenue_right` (from y).

Change it to something more descriptive:

```python
result = rsm.eda.combine(x, y, on="id", how="left", suffix="_after")
# x's "revenue" stays "revenue"; y's "revenue" becomes "revenue_after"
```

For binds and set operations, `suffix` is not used.

## 9. Plain-English interpretation templates

### Shape diagnostic

> Input: `x` is `<N_x>` × `<C_x>`; `y` is `<N_y>` × `<C_y>`. Using `how="<how>"`, the result is `<N>` × `<C>`. `<sanity check: did the row count match expectations?>`

### Inner join

> Inner join on `<key>`: keeps only the `<N>` rows in x with a matching key in y. `<N_x - N>` rows of x were dropped because they had no match. If those dropped rows matter, switch to `how="left"`.

### Left join

> Left join on `<key>`: keeps all `<N_x>` rows of x and attaches y-columns where there's a match. `<N_null>` x-rows have null y-columns (no match found). The unmatched rows are: `<list, or summarize>`.

### Full join

> Full join on `<key>`: keeps `<N>` rows from either side. `<N_x_only>` rows are x-only (null on the y side); `<N_y_only>` are y-only; `<N_matched>` are matched. If you only care about matched rows, switch to `inner`.

### Semi / anti

> Semi join: filters x to the `<N>` rows that have a y match (no y columns added). Effectively `x.filter(pl.col(key).is_in(y[key]))`.
> Anti join: keeps the `<N>` x-rows with NO match in y. Useful for finding the "lost" rows that an inner join would drop.

### Set operations

> Intersect: `<N>` rows appear in both datasets (as full row tuples).
> Union: `<N>` distinct rows across both datasets combined.
> Setdiff: `<N>` rows in x that don't appear in y.

## 10. Extending the output with polars

`rsm.eda.combine` returns a `pl.DataFrame`. Common next steps:

### Inspect what dropped (or got nulls)

```python
inner = rsm.eda.combine(x, y, on="id", how="inner")
left  = rsm.eda.combine(x, y, on="id", how="left")
print(f"Inner kept {inner.height} of x's {x.height} rows")
unmatched = left.filter(pl.col(y_only_col).is_null())
print(f"Unmatched x-rows: {unmatched.height}")
```

### Chain a transform

```python
result = (
    rsm.eda.combine(x, y, on="id", how="left")
    .with_columns(
        full_revenue=pl.col("revenue_x").fill_null(0) + pl.col("revenue_y").fill_null(0)
    )
)
```

### Chain to other pyrsm.eda functions

```python
# Join, then summarize
result = rsm.eda.combine(x, y, on="id", how="left")
stats = rsm.eda.explore(result, cols=["amount"], by="customer_segment")

# Join, then pivot
crosstab = rsm.eda.pivot(result, rows="region", cols="product", values="amount", agg="sum")

# Join, then plot
rsm.eda.visualize(result, x="date", y="amount", color="product", geom="line")
```

### Three-way join

`combine` is two-table; chain calls:

```python
a_b   = rsm.eda.combine(a, b, on="key", how="left")
a_b_c = rsm.eda.combine(a_b, c, on="key", how="left")
```

## 11. Worked examples (superheroes / publishers / avengers)

### Inner join — only complete matches

```python
import polars as pl
import pyrsm as rsm

superheroes = pl.read_parquet("<abs-path>/superheroes.parquet")
publishers = pl.read_parquet("<abs-path>/publishers.parquet")

rsm.eda.combine(superheroes, publishers, on="publisher", how="inner")
# 6 rows — Hellboy is dropped because "Dark Horse Comics" is not in publishers
```

### Left join — keep all superheroes

```python
rsm.eda.combine(superheroes, publishers, on="publisher", how="left")
# 7 rows — Hellboy retained with yr_founded = null
```

### Full join — keep everything from both

```python
rsm.eda.combine(superheroes, publishers, on="publisher", how="full")
# 8 rows — adds an "Image" row (publisher with no superhero), Hellboy retained
```

### Anti join — find the broken link

```python
rsm.eda.combine(superheroes, publishers, on="publisher", how="anti")
# 1 row: Hellboy (the only superhero whose publisher is missing)
```

Excellent diagnostic: if your inner join lost rows, run `anti` to see exactly which rows weren't matched.

### Semi join — filter superheroes to known publishers

```python
rsm.eda.combine(superheroes, publishers, on="publisher", how="semi")
# 6 rows — same as inner, but no publisher columns added (filter only)
```

### Bind rows — stack superheroes and avengers

```python
avengers = pl.read_parquet("<abs-path>/avengers.parquet")
combined = rsm.eda.combine(superheroes, avengers, how="bind_rows")
# 14 rows (7 + 7); Magneto appears twice (in both source tables)
```

### Union — deduplicate after binding

```python
rsm.eda.combine(superheroes, avengers, how="union")
# 13 rows — Magneto deduplicated
```

### Intersect — rows in both

```python
rsm.eda.combine(superheroes, avengers, how="intersect")
# 1 row: Magneto (the only character in both)
```

### Setdiff — rows in superheroes but not avengers

```python
rsm.eda.combine(superheroes, avengers, how="setdiff")
# 6 rows: all 6 non-Magneto superheroes
```

### Different column names

```python
x = pl.DataFrame({"customer_id": [1, 2, 3], "value": [10, 20, 30]})
y = pl.DataFrame({"cust_id": [1, 2, 4], "label": ["a", "b", "c"]})

rsm.eda.combine(x, y, left_on="customer_id", right_on="cust_id", how="left")
# Keeps x's customer_id; y's cust_id is dropped (used only for matching)
```

### Add — limit y's columns

```python
big_x = pl.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30]})
big_y = pl.DataFrame({"id": [1, 2, 3], "a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

rsm.eda.combine(big_x, big_y, on="id", how="left", add=["a"])
# Output has id, x, a — but not b, c
```

### Bind rows with partial column overlap

```python
df_a = pl.DataFrame({"id": [1, 2], "x": [10, 20]})
df_b = pl.DataFrame({"id": [3], "y": [30]})
rsm.eda.combine(df_a, df_b, how="bind_rows")
# 3 rows × 3 cols (id, x, y) — missing columns become null
```

## 12. Common pitfalls

- **Defaulting to `inner` silently drops rows.** Always state the expected row count before joining; verify after. Use `left` + null inspection when in doubt.
- **Duplicate keys on the y side multiply rows.** `y[key].n_unique() == y.height` is a good pre-flight check.
- **Swapping x and y reverses the meaning of left and right.** "Left join customer → orders" is different from "left join orders → customer". Be explicit about which side is the primary.
- **Joining on a wrong key**. If the keys mean different things in each table (e.g., one is `customer_id`, the other is `account_id` which is a different concept), the join is a bug even if it executes. Read the sidecar descriptions.
- **`bind_cols` without row alignment.** Positional concat. Only safe when row i of x corresponds to row i of y in some meaningful way.
- **Forgetting that `bind_rows` does NOT deduplicate.** Use `union` for that.
- **Forgetting that set operations compare entire rows.** `intersect`/`union`/`setdiff` use all columns. If the two tables have different columns, those operations will fail or behave unexpectedly.
- **Dtype mismatch on the key cascading into the output.** Auto-alignment can produce a different output dtype than either input. Note this if downstream code is dtype-sensitive.
- **Three-way join in one call.** Not supported; chain two-way calls.
- **Treating `combine` as a transform when a `polars` expression would do.** For derived columns, filtering, or sorting, use `.with_columns()` / `.filter()` / `.sort()`. `combine` is specifically for table-level operations.
- **Forgetting `add=` when joining a wide lookup table.** Adds all y columns by default; pass `add=[...]` to keep only what you need.
- **Using `suffix=` only when there's overlap.** If no non-key columns overlap, `suffix` does nothing. Don't confuse "specifying a suffix" with "renaming a column".
