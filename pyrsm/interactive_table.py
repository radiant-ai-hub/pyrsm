"""Interactive tables for notebooks and HTML reports (radiant.data's ``dtab``).

``rsm.dtab(df)`` shows a data frame as the table Radiant's Explore, Pivot and
Design results use: filter boxes above the columns (numbers take the Data >
View shorthand ``5..10``, ``..5``, ``>5``; text matches anywhere; a column with
a few distinct values gets a pick list), sorting, 10 rows a page (5 / 10 / 25 /
50 / All), a row count, "Clear filters" and "Download CSV". Tabulator ships
with pyrsm, so a self-contained HTML report works offline.

Where scripts do not run (a sanitized preview, a PDF or Word render), the same
table appears as a static one: the first page as HTML, or a Markdown table.
"""

from __future__ import annotations

import html
import json
import math
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

_WWW = Path(__file__).resolve().parent / "www"

#: Process-wide: has this kernel already sent Tabulator to the page? A report
#: renders one notebook in one fresh kernel, so the first table carries the
#: library and the rest wait for it.
_LIB_SENT = False

#: The look of Radiant's ResultTable (its global styles), plus the static
#: fallback and the row-count bar.
_CSS = """
.rsm-dtab { margin: 0.25rem 0 1rem; font-size: 0.9rem; }
.rsm-dtab .rsm-result-table__bar { display: flex; align-items: baseline; gap: 0.75rem; margin-bottom: 0.25rem;
  font-size: 0.8rem; color: #6c757d; }
.rsm-dtab .rsm-result-table__bar .rsm-dtab-actions { margin-left: auto; display: flex; gap: 1rem; }
.rsm-dtab .rsm-result-table__bar button { background: none; border: 0; padding: 0; color: #0d6efd;
  text-decoration: underline; font-size: 0.8rem; cursor: pointer; }
.rsm-dtab:not(.rsm-dtab--live) .rsm-result-table__bar .rsm-dtab-actions { display: none; }
.rsm-result-table__mount.tabulator { font-size: 0.85rem; border: 1px solid var(--bs-border-color, #dee2e6);
  max-width: 100%; }
.rsm-result-table__mount .tabulator-header-filter input,
.rsm-result-table__mount .tabulator-header-filter select { font-size: 0.75rem; padding: 0.1rem 0.25rem; }
.rsm-result-table__mount .tabulator-header-filter input::placeholder { color: var(--bs-secondary-color, #6c757d);
  opacity: 0.7; }
.rsm-result-table__mount .tabulator-cell.rsm-rt-label { background: var(--bs-secondary-bg, #e9ecef);
  font-weight: 600; }
.rsm-result-table__mount .tabulator-cell.rsm-rt-bold,
.rsm-result-table__mount .tabulator-calcs-bottom .tabulator-cell { font-weight: 700; }
.rsm-result-table__mount .tabulator-footer { font-size: 0.8rem; }
.rsm-result-table__mount .tabulator-footer .tabulator-footer-contents { gap: 0.75rem; flex-wrap: wrap; }
.rsm-dtab-static { border-collapse: collapse; font-size: 0.85rem; font-variant-numeric: tabular-nums; }
.rsm-dtab-static th, .rsm-dtab-static td { padding: 0.25rem 0.6rem; border-bottom: 1px solid #dee2e6;
  white-space: nowrap; background: transparent; }
.rsm-dtab-static thead th { border-bottom: 2px solid #495057; }
.rsm-dtab-static .rsm-num { text-align: right; }
.rsm-dtab-static .rsm-rt-label { background: #e9ecef; font-weight: 600; }
.rsm-dtab-static .rsm-rt-bold, .rsm-dtab-static tfoot td { font-weight: 700; }
"""


@lru_cache(maxsize=1)
def _assets() -> str:
    """Tabulator (MIT, see www/TABULATOR-LICENSE), its Bootstrap 5 theme and dtab.js."""
    tabulator = (_WWW / "js" / "tabulator.min.js").read_text(encoding="utf-8")
    theme = (_WWW / "tabulator_bootstrap5.min.css").read_text(encoding="utf-8")
    runner = (_WWW / "js" / "dtab.js").read_text(encoding="utf-8")
    return (
        f"<style>{theme}{_CSS}</style>"
        # Quarto's HTML (and classic Jupyter) loads require.js; with an AMD
        # ``define`` around, Tabulator registers as a module instead of
        # ``window.Tabulator``, so hide it while the library loads.
        "<script>if(!window.Tabulator){var __rsmDefine=window.define;window.define=undefined;"
        f"try{{{tabulator}\n}}finally{{window.define=__rsmDefine;}}}}</script>"
        f"<script>{runner}</script>"
    )


def format_cell(value: Any, kind: str, dec: int) -> str:
    """One cell as the table shows it (resultTable.ts ``formatCell``): integers
    as integers, other numbers with ``dec`` decimals (a value too small for them
    keeps three significant digits), a proportion as a percentage."""
    if value is None or value == "":
        return ""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Inf" if value > 0 else "-Inf")
    if kind == "percent":
        return f"{value * 100:.{dec}f}%"
    if float(value).is_integer():
        return str(int(value))
    if 0 < abs(value) < 10 ** -dec:
        text = f"{value:.3g}"
        return text.replace("e-0", "e-").replace("e+0", "e+")
    return f"{value:.{dec}f}"


def _kind(dtype: pl.DataType, name: str, percent: Sequence[str]) -> str:
    if name in percent:
        return "percent"
    return "number" if dtype.is_numeric() and dtype != pl.Boolean else "text"


def _cell(value: Any) -> Any:
    """A JSON-safe cell: numbers stay numbers, everything else is text."""
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Inf" if value > 0 else "-Inf")
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


class DTab:
    """An interactive table: ``_repr_html_`` for notebooks and HTML reports,
    ``_repr_markdown_`` for PDF and Word renders of a Quarto report."""

    def __init__(self, spec: Mapping[str, Any], frame: pl.DataFrame, caption: str = "") -> None:
        self.spec = dict(spec)
        self.frame = frame
        self.caption = caption

    # -- static versions -------------------------------------------------
    def _static_html(self) -> str:
        spec = self.spec
        dec, cols = spec["decimals"], spec["columns"]
        labels, bold = set(spec["labelColumns"]), set(spec["boldColumns"])

        def cls(c: Mapping[str, str]) -> str:
            out = ["rsm-num"] if c["kind"] != "text" else []
            out += ["rsm-rt-label"] if c["name"] in labels else []
            out += ["rsm-rt-bold"] if c["name"] in bold else []
            return f' class="{" ".join(out)}"' if out else ""

        def row_html(row: Mapping[str, Any], tag: str = "td") -> str:
            return "<tr>" + "".join(
                f"<{tag}{cls(c)}>{html.escape(format_cell(row.get(c['name']), c['kind'], dec))}</{tag}>" for c in cols
            ) + "</tr>"

        head = "".join(
            f'<th{" class=\"rsm-num\"" if c["kind"] != "text" else ""}>{html.escape(c["name"])}</th>' for c in cols
        )
        body = "".join(row_html(r) for r in spec["rows"][: spec["pageSize"]])
        foot = f"<tfoot>{row_html(spec['totals'])}</tfoot>" if spec.get("totals") else ""
        return f'<table class="rsm-dtab-static"><thead><tr>{head}</tr></thead><tbody>{body}</tbody>{foot}</table>'

    def _repr_html_(self) -> str:
        global _LIB_SENT
        spec_json = json.dumps(self.spec, allow_nan=False).replace("</", "<\\/")
        n, total = len(self.spec["rows"]), self.spec.get("totalRows") or len(self.spec["rows"])
        shown = min(n, self.spec["pageSize"])
        count = f"{n:,} rows" if shown == n else f"the first {shown:,} of {n:,} rows"
        if total > n:
            count += f" (the first {n:,} of {total:,})"
        caption = f'<div class="rsm-dtab-caption"><strong>{html.escape(self.caption)}</strong></div>' if self.caption else ""
        lib = ""
        if not _LIB_SENT:
            lib = _assets()
            _LIB_SENT = True
        uid = uuid.uuid4().hex[:10]
        # One wrapping element: a sanitized preview (DOMPurify) drops a <style>
        # that opens a fragment, and the scripts, and keeps the static table.
        return (
            f"<div class=\"rsm-dtab-wrap\">{lib}{caption}<div class=\"rsm-dtab\" id=\"rsm-dtab-{uid}\">"
            f'<div class="rsm-result-table__bar"><span class="rsm-dtab-count">{count}</span>'
            '<span class="rsm-dtab-actions"><button type="button" class="rsm-dtab-clear" style="display:none">'
            'Clear filters</button><button type="button" class="rsm-dtab-download">Download CSV</button></span></div>'
            f'<div class="rsm-result-table__mount">{self._static_html()}</div>'
            f'<script type="application/json" class="rsm-dtab-spec">{spec_json}</script>'
            "</div>"
            # Tabulator may still be loading (a later table than the one that
            # carries it); try again until it is there.
            "<script>(function(){var n=0;(function go(){if(window.rsmDtabRun&&window.Tabulator){window.rsmDtabRun();}"
            "else if(n++<100){setTimeout(go,100);}})();})();</script></div>"
        )

    def _repr_markdown_(self) -> str:
        from pyrsm.report import _table_md, column

        spec = self.spec
        nr = 20  # a printed page, not the whole table
        rows = spec["rows"][:nr] + ([spec["totals"]] if spec.get("totals") else [])
        labels = set(spec["labelColumns"])
        table = {
            "columns": [column(c["name"], c["name"], "term" if c["name"] in labels or c["kind"] == "text" else "number")
                        for c in spec["columns"]],
            "rows": [{c["name"]: format_cell(r.get(c["name"]), c["kind"], spec["decimals"]) for c in spec["columns"]}
                     for r in rows],
        }
        if self.caption:
            table["title"] = self.caption
        total = max(len(spec["rows"]), spec.get("totalRows") or 0)
        if total > nr:
            table["note"] = f"The first {nr} of {total:,} rows."
        return _table_md(table)

    def __repr__(self) -> str:
        return repr(self.frame)


def dtab(
    data: Any,
    dec: int = 3,
    nr: int | None = None,
    page_length: int = 10,
    label_columns: Sequence[str] | None = None,
    span: str | Mapping[str, Any] | None = None,
    totals: bool = False,
    bold_columns: Sequence[str] | None = None,
    format: str = "none",
    format_columns: Sequence[str] | None = None,
    percent: Sequence[str] = (),
    caption: str = "",
    filename: str = "table",
) -> DTab:
    """An interactive table of ``data`` for a notebook or an HTML report.

    Parameters
    ----------
    data : polars (or pandas) DataFrame
    dec : int
        Decimals for numbers that are not integers (and for percentages).
    nr : int, optional
        Rows to include (the first ``nr``); all rows by default. A report
        carries every included row, so keep large data short.
    page_length : int
        Rows per page (5, 10, 25, 50 or all can be picked in the table).
    label_columns : list of str, optional
        Row-label columns, shaded (an Explore or Pivot table's grouping variables).
    span : str or dict, optional
        A header over the value columns: a title (spanning every column that is
        not a label or "Total"), or ``{"title": ..., "columns": [...]}``.
    totals : bool
        Keep a last row whose label columns read "Total" (a pivot table's
        totals) below the table, outside sorting and filtering. Needs
        ``label_columns``.
    bold_columns : list of str, optional
        Columns in bold. Default: "Total", when there is one.
    format : {"none", "color_bar", "heat"}
        Pivot's conditional formatting of the value cells.
    format_columns : list of str, optional
        The cells to format. Default: the value columns other than "Total".
    percent : list of str
        Columns holding proportions, shown as percentages.
    caption : str
        A title above the table.
    filename : str
        File name (without .csv) for "Download CSV".

    Examples
    --------
    >>> rsm.dtab(diamonds, nr=100)
    >>> rsm.dtab(rsm.eda.explore(diamonds, cols=["price", "carat"], by="cut"),
    ...          label_columns=["cut", "variable"], span="Function")
    >>> rsm.dtab(rsm.eda.pivot(diamonds, rows="cut", cols="color", totals=True),
    ...          label_columns=["cut"], span="color", totals=True, format="heat")
    """
    frame = data if isinstance(data, pl.DataFrame) else pl.from_pandas(data) if hasattr(data, "to_numpy") else pl.DataFrame(data)
    total_rows = frame.height
    if nr is not None:
        frame = frame.head(int(nr))
    shown = frame  # the table as given (DTab.frame)
    # pivot(perc=True) cells ("23.40%") back to proportions shown as percentages,
    # so they sort and filter as numbers
    percent = list(percent)
    for c, t in frame.schema.items():
        if t == pl.String and c not in percent:
            vals = frame[c].drop_nulls()
            if len(vals) and vals.str.contains(r"^-?\d+(\.\d+)?%$").all():
                frame = frame.with_columns(pl.col(c).str.strip_suffix("%").cast(pl.Float64) / 100)
                percent.append(c)
    names = frame.columns
    label_columns = [c for c in (label_columns or []) if c in names]
    columns = [{"name": c, "kind": _kind(t, c, percent)} for c, t in frame.schema.items()]
    rows = [{k: _cell(v) for k, v in r.items()} for r in frame.iter_rows(named=True)]
    totals_row = None
    if totals and rows and label_columns and all(rows[-1].get(c) == "Total" for c in label_columns):
        totals_row, rows = rows[-1], rows[:-1]
    values = [c for c in names if c not in label_columns]
    if bold_columns is None:
        bold_columns = ["Total"] if "Total" in names else []
    if isinstance(span, str):
        span = {"title": span, "columns": [c for c in values if c != "Total"]}
    if format not in ("none", "color_bar", "heat"):
        raise ValueError('format must be "none", "color_bar" or "heat"')
    if format_columns is None:
        format_columns = [c for c in values if c != "Total"]
    spec = {
        "columns": columns,
        "rows": rows,
        "totals": totals_row,
        "decimals": int(dec),
        "labelColumns": list(label_columns),
        "boldColumns": list(bold_columns),
        "span": dict(span) if span else None,
        "conditional": format,
        "conditionalColumns": list(format_columns),
        "pageSize": int(page_length),
        # rows in the data, when only the first ``nr`` are in the table
        "totalRows": total_rows - (1 if totals_row else 0),
        "filename": filename,
    }
    return DTab(spec, shown, caption)
