"""Conjoint attributes with a few numeric values are categorical.

A conjoint design tests each attribute at a handful of levels, and a level
coded as a number (0/1 for a feature, 6/7/8 seats) is still a level: it gets
its own part-worth, with the lowest value as the base. ``conjoint`` and
``conjoint_choice`` treat a numeric attribute with at most ``max_levels``
distinct values (4 by default) this way; one with more values, and the price
attribute of a choice model, keeps a single slope.
"""

from __future__ import annotations

import polars as pl

__all__ = ["MAX_LEVELS", "few_valued", "level_text", "level_labels"]

#: Numeric attributes with at most this many distinct values are categorical.
MAX_LEVELS = 4


def level_text(df: pl.DataFrame, name: str) -> pl.Expr:
    """The column's values as level labels: 29 -> "29", 1.0 -> "1", 2.5 -> "2.5".

    A profile typed for prediction may hold the number as text ("29") or as a
    float (29.0); both get the same label. A value that is not a number keeps
    its text, so an unknown level is reported as typed.
    """
    dtype = df[name].dtype
    text = pl.col(name).cast(pl.Utf8)
    x = (pl.col(name) if dtype.is_numeric() else text).cast(pl.Float64, strict=False)
    whole = x.is_finite() & (x == x.round(0)) & (x.abs() < 2.0**53)
    number = (
        pl.when(whole)
        .then(x.cast(pl.Int64, strict=False).cast(pl.Utf8))
        .otherwise(x.cast(pl.Utf8))
    )
    return pl.coalesce(number, text).alias(name)


def level_labels(values: list, name: str = "x") -> list[str]:
    """``level_text`` for a list of values (e.g. the levels typed in ``cmd``)."""
    df = pl.DataFrame({name: values}, strict=False)
    return df.select(level_text(df, name))[name].to_list()


def few_valued(
    df: pl.DataFrame, evar: list[str], max_levels: int | None = MAX_LEVELS, keep=()
) -> dict[str, list[str]]:
    """Numeric attributes with at most ``max_levels`` distinct values, each with
    its level labels from the lowest value up. ``keep`` (the price) stays
    numeric; ``max_levels`` 0 or None keeps every numeric attribute numeric."""
    out: dict[str, list[str]] = {}
    if not max_levels or max_levels < 1:
        return out
    for v in evar:
        if v in keep or v in out or not df[v].dtype.is_numeric():
            continue
        values = df.select(pl.col(v).drop_nulls().unique().sort())
        if 0 < values.height <= max_levels:
            out[v] = values.select(level_text(values, v))[v].to_list()
    return out
