"""Formatted reports of pyrsm results: one layout for Radiant and for notebooks.

A report is a plain dict -- a header (title, ``label: value`` lines,
hypotheses), formatted tables, fit lines and optional sections -- in which every
cell is already a string. Radiant's result views draw it in the browser
(``PyrsmReport.svelte``); :func:`render_html` draws the same markup with the
same CSS, so ``summary(plain=False)`` in a notebook or a rendered Quarto report
shows the tables a student saw in Radiant.

* :func:`table` formats row dicts -- fixed decimals, integers as integers,
  ``< 0.001`` p-values, significance stars, ``Inf`` for an open interval;
* :func:`frame_table` does the same for a polars frame;
* :func:`text_header` / :func:`model_header` read the header pyrsm prints, and
  :func:`text_lines` picks lines such as a test's ``Chi-squared: ...`` result;
* :func:`build_report` assembles the report;
* :class:`Report` displays one: HTML in a notebook, plain text elsewhere.
"""

from __future__ import annotations

import contextlib
import html
import io
import math
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

SIG_CODES = "0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"

#: Footer for a table that carries significance stars.
SIG_NOTE = f"Signif. codes:  {SIG_CODES}"

#: Column keys that hold p-values (formatted ``< 0.001``) and stars.
P_VALUE_KEYS = frozenset({"p.value", "p.values", "p_value", "pvalue"})
STARS_KEYS = frozenset({"", " ", "  ", "stars", "sig"})


def capture(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()


def format_p_value(value: Any, dec: int = 3) -> str:
    """The one styled p-value shown by every report view (Chat card and Summary tab).

    Below ``10**-dec`` it reads ``< 0.001`` (at the default three decimals; an
    underflowed 0 included -- never a convincing "0.000"); otherwise fixed
    decimals, so the boundary 0.001 reads ``0.001``. Missing, non-finite or
    out-of-range values read ``n/a``. A string pyrsm already formatted (its
    ``< .001``) is mapped to the same presentation. Presentation only: the
    numbers, pyrsm's captured raw text and the generated code are untouched.
    """
    threshold = 10.0 ** -dec
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"<\s*0?\.0*1", text):
            return f"< {threshold:.{dec}f}"
        try:
            value = float(text)
        except ValueError:
            return "n/a"
    if isinstance(value, bool):
        return "n/a"
    try:
        p = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(p) or p < 0 or p > 1:
        return "n/a"
    if p < threshold:
        return f"< {threshold:.{dec}f}"
    return f"{p:.{dec}f}"


def model_header(model: Any) -> dict[str, Any]:
    """Parse the header pyrsm prints, so data scope lines come along too."""
    lines = [line.rstrip() for line in capture(model._summary_header).splitlines()]
    lines = [line for line in lines if line.strip()]
    title = lines[0] if lines else ""
    meta: list[dict[str, str]] = []
    hypotheses: list[dict[str, str]] = []
    for line in lines[1:]:
        if line.startswith(("Null hyp.", "Alt. hyp.")):
            label, _, value = line.partition(":")
            hypotheses.append({"label": label.strip(), "value": value.strip()})
            continue
        if line.strip().startswith("**") and line.strip().endswith("**"):
            # pyrsm's (and Radiant's) emphasised notes, e.g. "**Robust
            # standard errors used**": the view styles them, so drop the marks.
            meta.append({"label": "", "value": line.strip().strip("*").strip()})
            continue
        label, sep, value = line.partition(":")
        if sep:
            meta.append({"label": label.strip(), "value": value.strip()})
        else:
            meta.append({"label": "", "value": line.strip()})
    return {"title": title, "meta": meta, "hypotheses": hypotheses}


def number(value: Any, dec: int = 3) -> str:
    """One formatted cell: ints as ints, floats with ``dec`` fixed decimals,
    both with a comma as the thousands separator (1,234,567.891); a number
    pyrsm already printed as text gets the separator too.

    Non-finite numbers read as R prints them (``Inf``, ``-Inf``, ``NaN``), and
    so do the strings JSON stores them as (``"Inf"``, ``"NaN"``).
    Anything else is shown as text.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float) or hasattr(value, "dtype"):
        try:
            x = float(value)
        except (TypeError, ValueError):
            return str(value)
        if math.isnan(x):
            return "NaN"
        if math.isinf(x):
            return "Inf" if x > 0 else "-Inf"
        if hasattr(value, "dtype") and x.is_integer() and "int" in str(value.dtype):
            return f"{int(x):,}"
        return f"{round(x, dec) + 0.0:,.{dec}f}"  # never "-0.000"
    return _thousands(str(value).strip())


#: A plain number as text: sign, 4+ integer digits, optional decimals or %.
_PLAIN_NUMBER = re.compile(r"^(-?)([1-9]\d{3,})((?:\.\d+)?%?)$")


def _thousands(text: str) -> str:
    """``"1234567.5"`` -> ``"1,234,567.5"``; anything else unchanged."""
    m = _PLAIN_NUMBER.match(text)
    return f"{m.group(1)}{int(m.group(2)):,}{m.group(3)}" if m else text


def column(key: str, label: str | None = None, kind: str = "number") -> dict[str, str]:
    """A column of the rendered table: ``term`` (left, row label), ``number`` or ``stars``."""
    return {"key": key, "label": key if label is None else label, "kind": kind}


def table(
    id: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    title: str = "",
    subtitle: str = "",
    columns: Sequence[str | Mapping[str, str]] | None = None,
    terms: Sequence[str] = (),
    labels: Mapping[str, str] | None = None,
    dec: int = 3,
    note: str = "",
    percent: Sequence[str] = (),
) -> dict[str, Any]:
    """A report table from row dicts, every cell formatted here.

    ``columns`` fixes order (default: the keys of the first row). ``terms`` are
    row-label columns, drawn left-aligned. P-value keys get :func:`format_p_value`
    and a blank or ``stars`` key is the significance-stars column, always keyed
    ``stars`` in the output (its note is added automatically). ``labels`` renames headers without renaming keys.
    ``percent`` columns hold proportions shown as percentages (0.6348 -> 63.48%).
    """
    rows = [dict(r) for r in rows]
    labels = dict(labels or {})
    if columns is None:
        columns = list(rows[0]) if rows else []
    cols: list[dict[str, str]] = []
    for spec in columns:
        if isinstance(spec, Mapping):
            cols.append(dict(spec))
            continue
        key = str(spec)
        if key in STARS_KEYS:
            # one key for every tool (regression uses it too); read from `key`
            cols.append({**column("stars", "", "stars"), "source": key})
        elif key in terms:
            cols.append(column(key, labels.get(key, key), "term"))
        else:
            cols.append(column(key, labels.get(key, key)))
    out_rows = []
    for row in rows:
        out: dict[str, str] = {}
        for col in cols:
            key, value = col["key"], row.get(col.get("source", col["key"]))
            if col["kind"] == "stars":
                out[key] = "" if value is None else str(value).strip()
            elif key in P_VALUE_KEYS:
                out[key] = format_p_value(value, dec)
            elif col["kind"] == "term":
                out[key] = "" if value is None else str(value)
            elif key in percent and isinstance(value, (int, float)):
                out[key] = f"{100 * float(value):.{dec}f}%"
            else:
                out[key] = number(value, dec)
        out_rows.append(out)
    has_stars = any(c["kind"] == "stars" for c in cols)
    cols = [{k: v for k, v in c.items() if k != "source"} for c in cols]
    result: dict[str, Any] = {"id": id, "columns": cols, "rows": out_rows}
    if title:
        result["title"] = title
    if subtitle:
        result["subtitle"] = subtitle
    if note or has_stars:
        result["note"] = note or SIG_NOTE
    return result


def frame_table(
    id: str,
    frame: Any,
    *,
    terms: Sequence[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """:func:`table` from a polars DataFrame pyrsm returns.

    The first column is the row label unless ``terms`` says otherwise (pass
    ``terms=()`` for a frame with no label column).
    """
    columns = list(frame.columns)
    if terms is None:
        terms = columns[:1]
    return table(id, frame.iter_rows(named=True), columns=columns, terms=terms, **kwargs)


def text_lines(summary_text: str, *starts: str, contains: Sequence[str] = ()) -> list[str]:
    """Lines of pyrsm's text that start with ``starts`` (or contain ``contains``),
    e.g. a test's closing ``Chi-squared: ... p.value < .001`` line, with the
    p-value spelled as in the tables (``< 0.001``)."""
    out = []
    for line in (summary_text or "").splitlines():
        stripped = line.strip()
        if stripped and (stripped.startswith(starts) or any(c in stripped for c in contains)):
            out.append(re.sub(r"<\s*\.(0*1)\b", r"< 0.\1", stripped))
    return out


def text_header(summary_text: str) -> dict[str, Any]:
    """Title, ``label: value`` lines and hypotheses from pyrsm's printed header.

    pyrsm's plain summaries open with a title line and a block of
    ``Label : value`` lines (``Null hyp.`` / ``Alt. hyp.`` are kept apart), and
    a blank line separates blocks. Everything up to the first table is header;
    a line that is neither ``label: value`` nor blank ends it, and so does a
    ``Heading:`` with no value after a blank line.
    """
    lines = [line.rstrip() for line in (summary_text or "").splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    title = lines[0].strip() if lines else ""
    meta: list[dict[str, str]] = []
    hypotheses: list[dict[str, str]] = []
    after_blank = False
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            after_blank = True
            continue
        label, sep, value = stripped.partition(":")
        if not sep or not label.strip() or len(label) > 30 or label.strip().startswith(("P(", "1 - P(")):
            break
        # "Sample means:" after a blank line heads the next block, not the header
        # (a blank value inside the header block, e.g. "Upper bound :", is kept).
        if after_blank and not value.strip():
            break
        after_blank = False
        entry = {"label": label.strip(), "value": value.strip()}
        # pyrsm spells these "Null hyp." / "Null hyp" depending on the tool
        if entry["label"].rstrip(".") in ("Null hyp", "Alt. hyp"):
            entry["label"] = entry["label"].rstrip(".") + "."
            hypotheses.append(entry)
        else:
            meta.append(entry)
    return {"title": title, "meta": meta, "hypotheses": hypotheses}


def text_tables(summary_text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """The tables pyrsm printed (polars' boxed tables), and the other lines.

    Tools whose summaries are a run of labelled polars tables (the multivariate
    tools) get the shared report layout from their own printed output: each
    box becomes a :func:`table` titled by the ``Title:`` line above it, with the
    cells exactly as pyrsm formatted them. Returns ``(tables, lines)`` where
    ``lines`` are the remaining non-blank lines outside the header and tables
    (fit statistics such as ``Stress: 0.189``), in order.
    """
    lines = (summary_text or "").splitlines()
    header = text_header(summary_text)
    header_lines = 1 + len(header.get("meta", [])) + len(header.get("hypotheses", []))
    tables: list[dict[str, Any]] = []
    other: list[str] = []
    title = ""
    seen_header = 0
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if seen_header < header_lines:
            seen_header += 1
            i += 1
            continue
        # polars' shape line; the stars legend comes with the table's note
        if stripped.startswith(("shape: (", "Signif. codes")):
            i += 1
            continue
        if stripped.startswith("┌"):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("└"):
                j += 1
            box = [ln.strip() for ln in lines[i + 1 : j] if ln.strip().startswith("│")]
            names = [c.strip() for c in box[0].strip("│").split("┆")] if box else []
            body_start = next(
                (k + 1 for k, ln in enumerate(lines[i + 1 : j]) if ln.strip().startswith("╞")), 0
            )
            body = [
                ln.strip() for ln in lines[i + 1 + body_start : j] if ln.strip().startswith("│")
            ] if body_start else box[1:]
            cells = [[c.strip() for c in ln.strip("│").split("┆")] for ln in body]
            last = len(names) - 1

            def is_stars(k: int) -> bool:
                return k == last and k > 0 and all(
                    set(r[k]) <= set("*. ") for r in cells if len(r) > k
                )

            # An unnamed column is a significance-stars column only when it
            # holds stars; any other unnamed column keeps a blank header.
            keys = [n if n else ("" if is_stars(k) else f"col{k}") for k, n in enumerate(names)]
            rows = [dict(zip(keys, r)) for r in cells]
            label = title.rstrip(":").strip()

            def is_number(text: str) -> bool:
                try:
                    float(text.replace(",", "").rstrip("%"))
                except ValueError:
                    return text in ("", "Inf", "-Inf", "NaN", "null")
                return True

            # Leading columns of text (Attributes, Levels) are row labels.
            # Leading columns that hold text (Attributes, Levels) are row labels;
            # the first all-number column ends them (factor scores have none).
            terms = []
            for k, key in enumerate(keys):
                if all(is_number(r[k]) for r in cells if len(r) > k):
                    break
                terms.append(key)
            tables.append(
                table(
                    f"t{len(tables) + 1}", rows, title=label, columns=keys,
                    terms=terms, labels={k: "" for k in keys if k.startswith("col")},
                )
            )
            title = ""
            i = j + 1
            continue
        if stripped.endswith(":"):
            if title:
                other.append(title)
            title = stripped
        else:
            if title:
                other.append(title)
                title = ""
            other.append(re.sub(r"<\s*\.(0*1)\b", r"< 0.\1", stripped))
        i += 1
    if title:
        other.append(title)
    return tables, other


def text_report(kind: str, summary_text: str) -> dict[str, Any]:
    """:func:`build_report` from pyrsm's printed header, tables and fit lines."""
    tables, fit = text_tables(summary_text)
    return build_report(kind, header=text_header(summary_text), tables=tables, fit=fit)


def build_report(
    kind: str,
    *,
    header: Mapping[str, Any],
    tables: Sequence[Mapping[str, Any]] = (),
    fit: Sequence[str] = (),
    sections: Sequence[Mapping[str, Any]] = (),
    downloads: bool = True,
) -> dict[str, Any]:
    """The report ``PyrsmReport.svelte`` renders, with each table's row count.

    ``downloads`` puts a "Download CSV" link under each table in the tool
    workspace. Radiant offers those for tables a student takes elsewhere
    (coordinates, loadings, part-worths), not for test output, so the Basics
    tools turn it off.
    """
    report: dict[str, Any] = {
        "kind": kind,
        "title": header.get("title", ""),
        "meta": list(header.get("meta", [])),
        "hypotheses": list(header.get("hypotheses", [])),
        "tables": [dict(t) for t in tables],
        "fit": list(fit),
        "sections": [dict(s) for s in sections],
        "downloads": downloads,
    }
    every = report["tables"] + [s["table"] for s in report["sections"] if isinstance(s.get("table"), dict)]
    for t in every:
        t["total_rows"] = len(t.get("rows") or [])
    return report


def rows_table(id: str, rows: Sequence[Mapping[str, Any]], *, title: str = "", dec: int = 3,
               percent: Sequence[str] = ()) -> dict[str, Any]:
    """:func:`table` from plain row dicts; the leading text columns are row labels."""
    rows = [dict(r) for r in rows]
    columns = list(rows[0]) if rows else []
    terms = []
    for key in columns:
        if any(isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool) for r in rows):
            break
        terms.append(key)
    return table(id, rows, title=title, columns=columns, terms=terms, dec=dec, percent=percent)


# ---------------------------------------------------------------------------
# HTML: the markup and CSS of Radiant's result view
# ---------------------------------------------------------------------------

#: The CSS of Radiant's ``ReportHeader.svelte`` and ``PyrsmReport.svelte``,
#: verbatim (Radiant's tests compare the two), so a report looks the same in a
#: notebook as in Radiant.
CSS = """
  .pyrsm-header {
    margin-bottom: 0.9rem;
    font-size: 0.9rem;
  }
  .pyrsm-header__title {
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 0.35rem;
  }
  .pyrsm-meta,
  .pyrsm-hypotheses {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 0.1rem 0.9rem;
    margin: 0;
  }
  .pyrsm-meta dt,
  .pyrsm-hypotheses dt {
    color: #6b7280;
    font-weight: 500;
  }
  .pyrsm-meta dd,
  .pyrsm-hypotheses dd {
    margin: 0;
  }
  .pyrsm-hypotheses {
    margin-top: 0.35rem;
    font-size: 0.85rem;
  }
  .pyrsm-hypotheses dd {
    font-style: italic;
  }
  .pyrsm-report {
    --pyrsm-rule: #1f2937;
    --pyrsm-line: #e5e7eb;
    --pyrsm-muted: #6b7280;
    font-size: 0.9rem;
    color: var(--bs-body-color, #111827);
  }

  /* great_tables' look, which pyrsm uses in notebooks: heavy rules framing
     the table, a light rule under the column labels, no vertical lines. */
  .pyrsm-table-wrap {
    overflow-x: auto;
    max-width: 100%;
  }
  .pyrsm-table {
    border-collapse: collapse;
    border-top: 2px solid var(--pyrsm-rule);
    border-bottom: 2px solid var(--pyrsm-rule);
    font-variant-numeric: tabular-nums;
    width: auto;
    min-width: 22rem;
    margin-bottom: 0;
  }
  .pyrsm-table caption {
    caption-side: top;
    text-align: left;
    padding: 0 0 0.35rem;
    color: inherit;
  }
  .pyrsm-table__title {
    display: block;
    font-weight: 700;
    font-size: 0.95rem;
  }
  .pyrsm-csv {
    font-size: 0.75rem;
  }
  .pyrsm-table__subtitle {
    display: block;
    color: var(--pyrsm-muted);
    font-size: 0.82rem;
  }
  /* The host site's brand CSS styles every table (width: 100%, a tinted
     background and a heavy bottom border on every th). Reset that here so the
     report keeps pyrsm's look wherever it is mounted. */
  .pyrsm-table th,
  .pyrsm-table td {
    padding: 0.28rem 0.75rem;
    white-space: nowrap;
    background: transparent;
    border-bottom: 0;
    color: inherit;
  }
  .pyrsm-table tbody th {
    font-weight: 500;
  }
  .pyrsm-table thead th {
    border-bottom: 1px solid var(--pyrsm-rule);
    font-weight: 600;
    font-size: 0.82rem;
    color: var(--pyrsm-muted);
  }
  .pyrsm-table tbody tr + tr > * {
    border-top: 1px solid var(--pyrsm-line);
  }
  .pyrsm-col--term {
    text-align: left;
    font-weight: 500;
    font-family: var(--bs-font-monospace, ui-monospace, monospace);
    font-size: 0.85rem;
  }
  .pyrsm-col--number {
    text-align: right;
  }
  /* crtree's indented tree: connectors in a muted tone, terminal nodes marked
     with the green of the tree plot's leaf boxes, inner nodes muted, and the
     prediction drawn as a bar behind its number. */
  /* Beats the table's nowrap, which would collapse the indentation. */
  .pyrsm-table .pyrsm-col--tree {
    white-space: pre;
  }
  .pyrsm-guide {
    color: #9ca3af;
  }
  .pyrsm-row--leaf .pyrsm-node::before {
    content: "";
    display: inline-block;
    width: 0.6em;
    height: 0.6em;
    margin-right: 0.4em;
    background: #9acd32;
    border: 1px solid #333;
    vertical-align: baseline;
  }
  .pyrsm-row--branch > * {
    color: var(--pyrsm-muted);
  }
  .pyrsm-bar__track {
    display: inline-block;
    width: 5rem;
    height: 0.7em;
    margin-right: 0.6rem;
    vertical-align: middle;
    background: rgba(107, 114, 128, 0.08);
    border-radius: 2px;
    overflow: hidden;
  }
  .pyrsm-bar {
    display: block;
    height: 100%;
    background: rgba(107, 114, 128, 0.35);
  }
  .pyrsm-row--leaf .pyrsm-bar {
    background: #9acd32;
  }
  .pyrsm-bar__value {
    display: inline-block;
    min-width: 4.2em;
  }
  .pyrsm-col--stars {
    text-align: left;
    font-weight: 700;
    padding-left: 0.2rem !important;
    min-width: 2.2rem;
  }
  .pyrsm-table-block + .pyrsm-table-block {
    margin-top: 0.9rem;
  }
  .pyrsm-note {
    color: var(--pyrsm-muted);
    font-size: 0.78rem;
    margin-top: 0.3rem;
    white-space: pre-wrap;
  }

  .pyrsm-fit {
    margin-top: 0.9rem;
    display: grid;
    gap: 0.25rem;
  }
  .pyrsm-fit__line {
    display: flex;
    flex-wrap: wrap;
    gap: 0.2rem 1.2rem;
  }
  .pyrsm-fit__label {
    color: var(--pyrsm-muted);
    margin-right: 0.35rem;
  }
  .pyrsm-fit__value {
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }

  .pyrsm-section {
    margin-top: 1rem;
  }
  .pyrsm-section__title,
  .pyrsm-section > summary {
    font-weight: 700;
    font-size: 0.9rem;
    margin-bottom: 0.35rem;
  }
  details.pyrsm-section > summary {
    cursor: pointer;
  }
  .pyrsm-pre {
    font-family: var(--bs-font-monospace);
    font-size: 0.78rem;
    overflow-x: auto;
    white-space: pre;
    margin: 0;
  }
  .pyrsm-lines {
    font-variant-numeric: tabular-nums;
    display: grid;
    gap: 0.15rem;
  }
  .pyrsm-text {
    margin-top: 1rem;
  }
  .pyrsm-text > summary {
    font-size: 0.8rem;
    color: var(--pyrsm-muted);
    cursor: pointer;
  }
  .pyrsm-text pre {
    margin-top: 0.4rem;
    padding: 0.5rem;
    border: 1px solid var(--pyrsm-line);
    border-radius: 0.35rem;
    background: var(--bs-tertiary-bg, #f8f9fa);
    font-size: 0.78rem;
  }
"""

#: Notebook and Quarto pages style every table (stripes, borders, right
#: alignment); keep the report's own look there.
HOST_CSS = """
.pyrsm-report table.pyrsm-table { width: auto; margin: 0; background: transparent; }
.pyrsm-report table.pyrsm-table tr { background: transparent; }
.pyrsm-report table.pyrsm-table th, .pyrsm-report table.pyrsm-table td {
  border-left: 0; border-right: 0; background: transparent; line-height: 1.5; vertical-align: top;
}
.pyrsm-report table.pyrsm-table thead th { text-align: inherit; }
.pyrsm-report table.pyrsm-table thead th.pyrsm-col--number { text-align: right; }
.pyrsm-report table.pyrsm-table th.pyrsm-col--term { text-align: left; }
.pyrsm-report table.pyrsm-table td.pyrsm-col--number { text-align: right; }
.pyrsm-report table.pyrsm-table td.pyrsm-col--stars { text-align: left; }
.pyrsm-report dl, .pyrsm-report dd { margin-bottom: 0; }
"""


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _fit_parts(line: str) -> list[tuple[str, str]]:
    """``"R-squared: 0.925, Adjusted R-squared: 0.924"`` -> label/value pairs.

    Splits on ``", "`` only where the next piece starts a new ``Label:``, so
    ``df(18, 2981)`` stays intact (as ``PyrsmReport.svelte`` does).
    """
    parts = []
    for piece in re.split(r",\s+(?=[A-Za-z][^:,()]*:)", line):
        label, sep, value = piece.partition(":")
        parts.append((label.strip(), value.strip()) if sep else ("", piece.strip()))
    return parts


def _table_html(t: Mapping[str, Any], with_title: bool = True) -> str:
    out = ['<div class="pyrsm-table-wrap"><table class="pyrsm-table">']
    if with_title and t.get("title"):
        out.append(f'<caption><span class="pyrsm-table__title">{_esc(t["title"])}</span>')
        if t.get("subtitle"):
            out.append(f'<span class="pyrsm-table__subtitle">{_esc(t["subtitle"])}</span>')
        out.append("</caption>")
    columns = list(t.get("columns") or [])
    out.append("<thead><tr>")
    out.extend(f'<th class="pyrsm-col--{_esc(c.get("kind", "number"))}" scope="col">{_esc(c.get("label", ""))}</th>'
               for c in columns)
    out.append("</tr></thead><tbody>")
    for row in t.get("rows") or []:
        leaf = row.get("_leaf")
        cls = ' class="pyrsm-row--leaf"' if leaf == "1" else ' class="pyrsm-row--branch"' if leaf == "0" else ""
        out.append(f"<tr{cls}>")
        for c in columns:
            key, kind = c.get("key", ""), c.get("kind", "number")
            value = _esc(row.get(key, ""))
            if kind == "term":
                out.append(f'<th scope="row" class="pyrsm-col--term">{value}</th>')
            elif kind == "tree":
                out.append(f'<th scope="row" class="pyrsm-col--term pyrsm-col--tree"><span class="pyrsm-guide">'
                           f'{_esc(row.get("_guide", ""))}</span><span class="pyrsm-node">{value}</span></th>')
            elif kind == "bar":
                try:
                    width = 100 * float(row.get(f"{key}__bar") or 0)
                except (TypeError, ValueError):
                    width = 0
                out.append(f'<td class="pyrsm-col--number pyrsm-col--bar"><span class="pyrsm-bar__track">'
                           f'<span class="pyrsm-bar" style="width: {width:g}%"></span></span>'
                           f'<span class="pyrsm-bar__value">{value}</span></td>')
            else:
                out.append(f'<td class="pyrsm-col--{_esc(kind)}">{value}</td>')
        out.append("</tr>")
    out.append("</tbody></table></div>")
    if t.get("note"):
        out.append(f'<div class="pyrsm-note">{_esc(t["note"])}</div>')
    return "".join(out)


def _section_html(section: Mapping[str, Any]) -> str:
    out = []
    if isinstance(section.get("table"), Mapping):
        out.append(_table_html(section["table"], with_title=False))
    lines = section.get("lines")
    if lines and section.get("preformatted"):
        out.append(f'<pre class="pyrsm-pre">{_esc(chr(10).join(lines))}</pre>')
    elif lines:
        out.append('<div class="pyrsm-lines">' + "".join(f"<div>{_esc(x)}</div>" for x in lines) + "</div>")
    for t in section.get("tables") or []:
        out.append(f'<div class="pyrsm-table-block">{_table_html(t)}</div>')
    return "".join(out)


def render_html(report: Mapping[str, Any], *, style: bool = True) -> str:
    """The report as HTML: ``PyrsmReport.svelte``'s markup, with its CSS."""
    out = [f'<section class="pyrsm-report" aria-label="{_esc(report.get("title", ""))}">']
    if style:
        # Inside the section: an HTML sanitizer (DOMPurify, as Radiant's report
        # preview uses) drops a <style> that opens a fragment, not one inside it.
        out.append(f"<style>{CSS}{HOST_CSS}</style>")
    out.append('<header class="pyrsm-header">')
    if report.get("title"):
        out.append(f'<div class="pyrsm-header__title">{_esc(report["title"])}</div>')
    out.append('<dl class="pyrsm-meta">')
    out.extend(f"<dt>{_esc(m.get('label'))}</dt><dd>{_esc(m.get('value'))}</dd>" for m in report.get("meta") or [])
    out.append("</dl>")
    if report.get("hypotheses"):
        out.append('<dl class="pyrsm-hypotheses">')
        out.extend(f"<dt>{_esc(h.get('label'))}</dt><dd>{_esc(h.get('value'))}</dd>" for h in report["hypotheses"])
        out.append("</dl>")
    out.append("</header>")
    for t in report.get("tables") or []:
        out.append(f'<div class="pyrsm-table-block">{_table_html(t)}</div>')
    if report.get("fit"):
        out.append('<div class="pyrsm-fit" aria-label="Model fit">')
        for line in report["fit"]:
            out.append('<div class="pyrsm-fit__line">')
            for label, value in _fit_parts(line):
                out.append('<span class="pyrsm-fit__part">')
                if label:
                    out.append(f'<span class="pyrsm-fit__label">{_esc(label)}</span>')
                out.append(f'<span class="pyrsm-fit__value">{_esc(value)}</span></span>')
            out.append("</div>")
        out.append("</div>")
    for section in report.get("sections") or []:
        if section.get("collapsed"):
            out.append(f'<details class="pyrsm-section"><summary>{_esc(section.get("title"))}</summary>'
                       f"{_section_html(section)}</details>")
        else:
            out.append(f'<div class="pyrsm-section"><div class="pyrsm-section__title">{_esc(section.get("title"))}'
                       f"</div>{_section_html(section)}</div>")
    out.append("</section>")
    return "".join(out)


def _md(value: Any) -> str:
    """Text safe inside Markdown (and a pipe-table cell)."""
    text = "" if value is None else str(value)
    return re.sub(r"([\\`*_\[\]<>|#])", r"\\\1", text)


def _table_md(t: Mapping[str, Any], with_title: bool = True) -> str:
    columns = list(t.get("columns") or [])
    if not columns:
        return ""
    out = []
    if with_title and t.get("title"):
        sub = f" -- {_md(t['subtitle'])}" if t.get("subtitle") else ""
        out.append(f"**{_md(t['title'])}**{sub}\n")
    rows = []
    for row in t.get("rows") or []:
        cells = []
        for c in columns:
            value = row.get(c.get("key", ""), "")
            if c.get("kind") == "tree":
                value = f"{row.get('_guide', '')}{value}"
            text = "" if value is None else str(value)
            if c.get("kind") not in ("term", "tree") and re.match(r"-[0-9.]", text):
                # a true minus: LaTeX would break a line after a hyphen
                text = "\u2212" + text[1:]
            cells.append(text)
        rows.append(cells)
    labels = [str(c.get("label", "")) for c in columns]
    # A wide table wraps to the page; pandoc sizes its columns by the dashes
    # under the header. Numbers get their full width (never broken), while
    # long labels ("AsstProf not equal to AssocProf") wrap instead.
    rule = []
    for j, c in enumerate(columns):
        width = max([len(labels[j])] + [len(r[j]) for r in rows]) + 1
        if c.get("kind") in ("term", "tree"):
            rule.append(":" + "-" * max(3, min(width, 10)))
        elif c.get("kind") == "stars":
            rule.append(":" + "-" * 3)
        else:
            rule.append("-" * max(3, width) + ":")
    lines = ["| " + " | ".join(_md(x) or " " for x in labels) + " |", "|" + "|".join(rule) + "|"]
    lines += ["| " + " | ".join(_md(x) or " " for x in r) + " |" for r in rows]
    if max(len(line) for line in lines) > 72:
        # Too wide for a PDF page at the body font: a smaller font there (Word
        # and HTML ignore a LaTeX block), so the numbers keep their columns.
        lines = ["```{=latex}", "\\begingroup\\footnotesize", "```", "", *lines,
                 "", "```{=latex}", "\\endgroup", "```"]
    out.extend(lines)
    if t.get("note"):
        out.append(f"\n{_md(t['note'])}")
    return "\n".join(out) + "\n"


def render_markdown(report: Mapping[str, Any]) -> str:
    """The report as Markdown: what a PDF or Word render of a Quarto report shows.

    Quarto uses a result's HTML for HTML output and its Markdown for PDF and
    Word, where pandoc turns the pipe tables into native tables that fit the page.
    """
    out = []
    if report.get("title"):
        out.append(f"**{_md(report['title'])}**\n")
    lines = [f"{_md(m.get('label'))}: {_md(m.get('value'))}" if m.get("label") else _md(m.get("value"))
             for m in report.get("meta") or []]
    lines += [f"{_md(h.get('label'))}: *{_md(h.get('value'))}*" for h in report.get("hypotheses") or []]
    if lines:
        out.append("\\\n".join(lines) + "\n")
    for t in report.get("tables") or []:
        out.append(_table_md(t))
    if report.get("fit"):
        out.append("\\\n".join(_md(line) for line in report["fit"]) + "\n")
    for section in report.get("sections") or []:
        out.append(f"**{_md(section.get('title'))}**\n")
        if isinstance(section.get("table"), Mapping):
            out.append(_table_md(section["table"], with_title=False))
        if section.get("lines") and section.get("preformatted"):
            out.append("```\n" + "\n".join(section["lines"]) + "\n```\n")
        elif section.get("lines"):
            out.append("\\\n".join(_md(x) for x in section["lines"]) + "\n")
        for t in section.get("tables") or []:
            out.append(_table_md(t))
    return "\n".join(out)


class Report:
    """A report to display: Radiant's HTML in a notebook, Markdown for a PDF or
    Word render of a Quarto report, pyrsm's text elsewhere.

    ``summary(plain=False)`` displays one in a notebook (and so in a rendered
    Quarto report). ``data`` is the report dict; ``text`` the plain summary.
    """

    def __init__(self, data: Mapping[str, Any], text: str = "") -> None:
        self.data = dict(data)
        self.text = text

    def _repr_html_(self) -> str:
        return render_html(self.data)

    def _repr_markdown_(self) -> str:
        return render_markdown(self.data)

    def _repr_pretty_(self, printer: Any, cycle: bool) -> None:
        printer.text(self.text or str(self.data.get("title", "")))

    def __repr__(self) -> str:
        return self.text or f"Report({self.data.get('title', '')!r})"


def display(report: Mapping[str, Any], text: str = "") -> None:
    """Show ``report`` in a notebook (IPython's display)."""
    from IPython.display import display as ipy_display

    ipy_display(Report(report, text))
