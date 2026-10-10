"""Detailed, structured comparison of two data frames.

``compare_frames(expected, actual)`` answers "are these the same data?" the way a
grader does -- strict equality -- and, when they are not, says *why* in terms a
student can act on: a missing column, a column in the wrong place, ``Int64``
where ``Categorical`` was expected, 14 values that differ in ``first``, duplicate
or unmatched join keys.

Nothing is coerced into passing: dtype differences are reported, values are
compared as text only to tell "same values, wrong type" apart from "different
values", and ``equal`` is ``expected.equals(actual)`` and nothing else.

Pandas frames are accepted and converted with ``pl.from_pandas``; the result
records that a conversion happened.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import polars as pl

__all__ = ["ColumnDiff", "FrameComparison", "KeyDiagnostics", "compare_frames"]


@dataclass
class ColumnDiff:
    name: str
    in_expected: bool
    in_actual: bool
    position_expected: int | None = None
    position_actual: int | None = None
    dtype_expected: str | None = None
    dtype_actual: str | None = None
    dtype_equal: bool = False
    nulls_expected: int | None = None
    nulls_actual: int | None = None
    #: Rows whose values differ (null vs value counts as a difference). ``None``
    #: when values could not be compared (column missing, or rows not alignable).
    value_mismatches: int | None = None
    #: The values are equal once both sides are read as text: the type is wrong,
    #: the data is not.
    equal_as_text: bool | None = None
    examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class KeyDiagnostics:
    key: list[str]
    duplicates_expected: int = 0
    duplicates_actual: int = 0
    unmatched_expected: int = 0
    unmatched_actual: int = 0
    examples_duplicate_actual: list[Any] = field(default_factory=list)
    examples_unmatched_expected: list[Any] = field(default_factory=list)
    examples_unmatched_actual: list[Any] = field(default_factory=list)


@dataclass
class FrameComparison:
    #: Strict: same column names in the same order, same dtypes, and
    #: ``expected.equals(actual)``.
    equal: bool
    shape_expected: tuple[int, int]
    shape_actual: tuple[int, int]
    column_order_equal: bool
    missing_columns: list[str]
    extra_columns: list[str]
    columns: list[ColumnDiff]
    key: KeyDiagnostics | None = None
    aligned_by: str = "position"
    converted_from_pandas: bool = False
    #: ``expected.equals(actual)`` alone. Polars' ``equals`` does not compare
    #: dtypes (Int32 equals Int64), so this can be True while ``equal`` is False.
    polars_equals: bool = False
    messages: list[str] = field(default_factory=list)

    @property
    def dtype_mismatches(self) -> list[ColumnDiff]:
        return [c for c in self.columns if c.in_expected and c.in_actual and not c.dtype_equal]

    @property
    def value_mismatches(self) -> list[ColumnDiff]:
        return [c for c in self.columns if c.value_mismatches]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["shape_expected"] = list(self.shape_expected)
        out["shape_actual"] = list(self.shape_actual)
        return out

    def summary(self) -> str:
        if self.equal:
            return "The data frames are identical (strict equality)."
        return "\n".join(self.messages) or "The data frames differ."


def _to_polars(frame: Any) -> tuple[pl.DataFrame, bool]:
    if isinstance(frame, pl.DataFrame):
        return frame, False
    if isinstance(frame, pl.LazyFrame):
        return frame.collect(), False
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a pyrsm dependency
        pd = None
    if pd is not None and isinstance(frame, pd.DataFrame):
        return pl.from_pandas(frame), True
    raise TypeError(f"expected a Polars or Pandas DataFrame, got {type(frame).__name__}")


def _as_text(series: pl.Series) -> pl.Series:
    if series.dtype == pl.String:
        return series
    try:
        return series.cast(pl.String)
    except Exception:  # noqa: BLE001 - nested/object types
        return pl.Series(series.name, [None if v is None else str(v) for v in series.to_list()], dtype=pl.String)


def _comparable(series: pl.Series) -> pl.Series:
    # Categorical/Enum columns from different frames may not share a category
    # mapping; compare their labels instead.
    if series.dtype in (pl.Categorical,) or isinstance(series.dtype, (pl.Categorical, pl.Enum)):
        return series.cast(pl.String)
    return series


def _differs(a: pl.Series, b: pl.Series) -> pl.Series:
    """Row-wise "not equal", treating two nulls as equal."""
    return ~a.eq_missing(b).fill_null(False)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def compare_frames(
    expected: Any,
    actual: Any,
    *,
    key: str | list[str] | None = None,
    strict: bool = True,
    max_examples: int = 5,
    show_values: bool = True,
) -> FrameComparison:
    """Compare ``actual`` against ``expected`` and explain every difference.

    Parameters
    ----------
    expected, actual
        Polars (or Pandas) data frames.
    key
        Column(s) identifying a row. When given, rows are matched on the key
        (and duplicate / unmatched keys are reported); otherwise rows are
        compared by position, which needs equal heights.
    strict
        ``True`` (the default): ``equal`` requires the same schema (names, order
        and dtypes) and ``expected.equals(actual)``. Polars' ``equals`` alone does
        not compare dtypes, so it is reported separately as ``polars_equals``.
        ``False`` relaxes only the column *order*. Values are never coerced.
    max_examples
        At most this many example rows per differing column.
    show_values
        Include the differing values in the examples (``False`` reports only the
        row position or key).
    """
    exp, conv_e = _to_polars(expected)
    act, conv_a = _to_polars(actual)
    keys = [key] if isinstance(key, str) else list(key or [])

    exp_cols, act_cols = list(exp.columns), list(act.columns)
    missing = [c for c in exp_cols if c not in act_cols]
    extra = [c for c in act_cols if c not in exp_cols]
    order_equal = exp_cols == act_cols

    messages: list[str] = []
    if exp.shape != act.shape:
        messages.append(
            f"Shape differs: expected {exp.height} rows x {exp.width} columns, "
            f"got {act.height} rows x {act.width} columns."
        )
    for c in missing:
        messages.append(f"Column `{c}` is missing.")
    for c in extra:
        messages.append(f"Column `{c}` is not in the expected data.")
    if not missing and not extra and not order_equal:
        moved = [c for i, c in enumerate(exp_cols) if act_cols[i] != c]
        messages.append(
            "Columns are in a different order (first out of place: "
            + ", ".join(f"`{c}`" for c in moved[:5])
            + ")."
        )

    # --- row alignment -------------------------------------------------------
    key_diag: KeyDiagnostics | None = None
    aligned_exp: pl.DataFrame | None = None
    aligned_act: pl.DataFrame | None = None
    aligned_by = "position"
    if keys:
        aligned_by = "key"
        absent = [k for k in keys if k not in exp_cols or k not in act_cols]
        if absent:
            messages.append("Key column(s) " + ", ".join(f"`{k}`" for k in absent) + " not present in both frames.")
        else:
            key_diag = KeyDiagnostics(key=keys)
            ke = exp.select([_comparable(exp[k]).cast(pl.String).alias(k) for k in keys])
            ka = act.select([_comparable(act[k]).cast(pl.String).alias(k) for k in keys])
            dup_e = ke.filter(ke.is_duplicated())
            dup_a = ka.filter(ka.is_duplicated())
            key_diag.duplicates_expected = dup_e.unique().height
            key_diag.duplicates_actual = dup_a.unique().height
            key_diag.examples_duplicate_actual = [_jsonable(r if len(r) > 1 else r[0]) for r in dup_a.unique().head(max_examples).rows()]
            only_e = ke.unique().join(ka.unique(), on=keys, how="anti")
            only_a = ka.unique().join(ke.unique(), on=keys, how="anti")
            key_diag.unmatched_expected = only_e.height
            key_diag.unmatched_actual = only_a.height
            key_diag.examples_unmatched_expected = [_jsonable(r if len(r) > 1 else r[0]) for r in only_e.head(max_examples).rows()]
            key_diag.examples_unmatched_actual = [_jsonable(r if len(r) > 1 else r[0]) for r in only_a.head(max_examples).rows()]
            label = ", ".join(f"`{k}`" for k in keys)
            if key_diag.duplicates_actual:
                messages.append(f"{key_diag.duplicates_actual} {label} value(s) appear more than once in the result.")
            if key_diag.duplicates_expected:
                messages.append(f"{key_diag.duplicates_expected} {label} value(s) are duplicated in the expected data.")
            if key_diag.unmatched_expected:
                messages.append(f"{key_diag.unmatched_expected} expected {label} value(s) are missing from the result.")
            if key_diag.unmatched_actual:
                messages.append(f"{key_diag.unmatched_actual} {label} value(s) in the result are not in the expected data.")
            if not key_diag.duplicates_actual and not key_diag.duplicates_expected:
                tag = "__pyrsm_row__"
                e_idx = exp.with_row_index(tag).with_columns([_comparable(exp[k]).cast(pl.String).alias(k) for k in keys])
                a_idx = act.with_row_index(tag).with_columns([_comparable(act[k]).cast(pl.String).alias(k) for k in keys])
                pairs = e_idx.select(keys + [tag]).join(a_idx.select(keys + [tag]), on=keys, how="inner", suffix="_a")
                aligned_exp = exp[pairs[tag].to_list()]
                aligned_act = act[pairs[f"{tag}_a"].to_list()]
                # keep the key labels for examples
                aligned_keys = pairs.select(keys)
    elif exp.height == act.height:
        aligned_exp, aligned_act = exp, act
    elif not messages or not messages[0].startswith("Shape differs"):
        messages.append("Row counts differ, so values were not compared by position (pass key=... to match rows).")

    # --- per column ------------------------------------------------------------
    columns: list[ColumnDiff] = []
    for name in exp_cols + extra:
        in_e, in_a = name in exp_cols, name in act_cols
        diff = ColumnDiff(
            name=name,
            in_expected=in_e,
            in_actual=in_a,
            position_expected=exp_cols.index(name) if in_e else None,
            position_actual=act_cols.index(name) if in_a else None,
            dtype_expected=str(exp[name].dtype) if in_e else None,
            dtype_actual=str(act[name].dtype) if in_a else None,
            nulls_expected=exp[name].null_count() if in_e else None,
            nulls_actual=act[name].null_count() if in_a else None,
        )
        diff.dtype_equal = bool(in_e and in_a and exp[name].dtype == act[name].dtype)
        if in_e and in_a and not diff.dtype_equal:
            messages.append(f"`{name}` is {diff.dtype_actual}, expected {diff.dtype_expected}.")
        if in_e and in_a and diff.nulls_expected != diff.nulls_actual:
            messages.append(f"`{name}` has {diff.nulls_actual} missing value(s), expected {diff.nulls_expected}.")
        if in_e and in_a and aligned_exp is not None and aligned_act is not None:
            se, sa = _comparable(aligned_exp[name]), _comparable(aligned_act[name])
            if se.dtype == sa.dtype:
                mask = _differs(se, sa)
            else:
                mask = _differs(_as_text(se), _as_text(sa))
            count = int(mask.sum())
            diff.value_mismatches = count
            if not diff.dtype_equal:
                diff.equal_as_text = count == 0
            if count:
                messages.append(f"`{name}`: {count} value(s) differ.")
                rows = mask.arg_true().head(max_examples).to_list()
                for r in rows:
                    ex: dict[str, Any] = {"row": int(r)}
                    if aligned_by == "key":
                        ex["key"] = _jsonable(aligned_keys.row(r) if len(keys) > 1 else aligned_keys.row(r)[0])
                    if show_values:
                        ex["expected"] = _jsonable(se[r])
                        ex["actual"] = _jsonable(sa[r])
                    diff.examples.append(ex)
        columns.append(diff)

    polars_equals = bool(exp.equals(act))
    equal = polars_equals and exp.schema == act.schema
    if not strict and not equal and not missing and not extra:
        reordered = act.select(exp_cols)
        equal = bool(exp.equals(reordered)) and exp.schema == reordered.schema
    if equal:
        messages = []
    elif not messages:
        messages.append("The frames differ in a way the column checks did not isolate (e.g. row order).")

    return FrameComparison(
        equal=equal,
        shape_expected=exp.shape,
        shape_actual=act.shape,
        column_order_equal=order_equal,
        missing_columns=missing,
        extra_columns=extra,
        columns=columns,
        key=key_diag,
        aligned_by=aligned_by,
        converted_from_pandas=conv_e or conv_a,
        polars_equals=polars_equals,
        messages=messages,
    )
