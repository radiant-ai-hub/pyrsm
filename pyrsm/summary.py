"""Small, JSON-safe analytical summary documents and semantic renderers.

The public contract in this module deliberately has no notebook or web-framework
dependency. Analytical classes choose their values and sections; consumers choose
their surrounding layout. That keeps a browser client from reverse-engineering a
table out of a printed console summary.
"""

from __future__ import annotations

import html
import math
from collections.abc import Iterable, Mapping
from typing import Any


def json_value(value: Any) -> Any:
    """Convert a scalar from pandas/Polars/NumPy to strict JSON data.

    ``NaN`` and infinities are represented as ``None`` rather than as JavaScript's
    non-standard numeric literals. An unsupported value is retained only as its
    safe string form; model/data objects are never passed to this function.
    """
    if hasattr(value, "item") and not isinstance(value, str | bytes):
        try:
            return json_value(value.item())
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    return str(value)


def table_rows(frame: Any, *, include_index: bool = False) -> list[dict[str, Any]]:
    """Turn a pandas or Polars table into JSON-safe row dictionaries."""
    if hasattr(frame, "to_dicts"):
        raw_rows = frame.to_dicts()
    elif hasattr(frame, "iterrows"):
        raw_rows = []
        for index, row in frame.iterrows():
            item = {str(key): value for key, value in row.to_dict().items()}
            if include_index:
                item = {"index": index, **item}
            raw_rows.append(item)
    else:
        raise TypeError("summary table must be a pandas or Polars dataframe")
    return [
        {str(key): json_value(value) for key, value in row.items()}
        for row in raw_rows
    ]


def column(
    key: str,
    *,
    label: str | None = None,
    semantic_type: str = "text",
    align: str | None = None,
    decimals: int | None = None,
) -> dict[str, Any]:
    """Describe one machine-readable summary-table column."""
    return {
        "key": key,
        "label": key if label is None else label,
        "semantic_type": semantic_type,
        "align": align or ("end" if semantic_type in {"number", "p_value", "percent"} else "start"),
        "format": {"decimals": decimals} if decimals is not None else {},
    }


def table_section(
    identifier: str,
    title: str,
    rows: Iterable[Mapping[str, Any]],
    columns: list[dict[str, Any]],
    *,
    footnotes: list[str] | None = None,
) -> dict[str, Any]:
    """Create one table section while retaining raw values for all rows."""
    return {
        "id": identifier,
        "title": title,
        "kind": "table",
        "columns": columns,
        "rows": [
            {str(key): json_value(value) for key, value in row.items()} for row in rows
        ],
        "values": [],
        "footnotes": footnotes or [],
    }


def key_value_section(
    identifier: str, title: str, values: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "id": identifier,
        "title": title,
        "kind": "key_value",
        "columns": [],
        "rows": [],
        "values": [
            {"label": str(key), "value": json_value(value)}
            for key, value in values.items()
        ],
        "footnotes": [],
    }


def document(
    *,
    producer: str,
    title: str,
    analysis_kind: str,
    options: Mapping[str, Any],
    metadata: Mapping[str, Any],
    sections: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the versioned public summary shape shared across pyrsm tools."""
    return {
        "schema_version": 1,
        "producer": producer,
        "title": title,
        "analysis_kind": analysis_kind,
        "options": {str(key): json_value(value) for key, value in options.items()},
        "metadata": [
            {"label": str(key), "value": json_value(value)}
            for key, value in metadata.items()
        ],
        "sections": sections,
    }


def _display(value: Any, column_spec: Mapping[str, Any] | None = None) -> str:
    if value is None:
        return "—"
    column_spec = column_spec or {}
    semantic = column_spec.get("semantic_type")
    decimals = (column_spec.get("format") or {}).get("decimals")
    if semantic == "p_value" and isinstance(value, int | float):
        if value < 0.001:
            return "< .001"
        return f"{value:.{decimals if decimals is not None else 3}f}"
    if isinstance(value, float):
        return f"{value:.{decimals if decimals is not None else 3}f}"
    return str(value)


def render_summary_text(summary: Mapping[str, Any]) -> str:
    """Render a complete, readable plain-text form of a summary document."""
    lines = [str(summary.get("title", "Analysis summary"))]
    for item in summary.get("metadata", []):
        lines.append(f"{item.get('label', '')}: {_display(item.get('value'))}")
    for section in summary.get("sections", []):
        lines.extend(["", str(section.get("title", ""))])
        if section.get("kind") == "key_value":
            lines.extend(
                f"{item.get('label', '')}: {_display(item.get('value'))}"
                for item in section.get("values", [])
            )
            continue
        columns = list(section.get("columns", []))
        if columns:
            lines.append(" | ".join(str(column.get("label", "")) for column in columns))
            lines.append(" | ".join("---" for _ in columns))
            for row in section.get("rows", []):
                lines.append(
                    " | ".join(
                        _display(row.get(column.get("key", "")), column)
                        for column in columns
                    )
                )
        lines.extend(str(note) for note in section.get("footnotes", []))
    return "\n".join(lines)


def render_summary_html(summary: Mapping[str, Any]) -> str:
    """Render semantic, escaped HTML with stable classes and no script."""
    parts = [
        '<section class="pyrsm-summary" aria-label="Analysis summary">',
        f'<h2 class="pyrsm-summary__title">{html.escape(str(summary.get("title", "Analysis summary")))}</h2>',
        '<dl class="pyrsm-summary__metadata">',
    ]
    for item in summary.get("metadata", []):
        parts.extend(
            [
                f'<dt>{html.escape(str(item.get("label", "")))}</dt>',
                f'<dd>{html.escape(_display(item.get("value")))}</dd>',
            ]
        )
    parts.append("</dl>")
    for section in summary.get("sections", []):
        title = html.escape(str(section.get("title", "")))
        parts.append(f'<section class="pyrsm-summary__section"><h3>{title}</h3>')
        if section.get("kind") == "key_value":
            parts.append('<dl class="pyrsm-summary__key-values">')
            for item in section.get("values", []):
                parts.extend(
                    [
                        f'<dt>{html.escape(str(item.get("label", "")))}</dt>',
                        f'<dd>{html.escape(_display(item.get("value")))}</dd>',
                    ]
                )
            parts.append("</dl>")
        else:
            columns = list(section.get("columns", []))
            parts.append('<div class="pyrsm-summary__table-wrap"><table>')
            parts.append("<thead><tr>")
            for spec in columns:
                parts.append(f'<th scope="col">{html.escape(str(spec.get("label", "")))}</th>')
            parts.append("</tr></thead><tbody>")
            for row in section.get("rows", []):
                parts.append("<tr>")
                for spec in columns:
                    value = _display(row.get(spec.get("key", "")), spec)
                    parts.append(f"<td>{html.escape(value)}</td>")
                parts.append("</tr>")
            parts.append("</tbody></table></div>")
        for note in section.get("footnotes", []):
            parts.append(f'<p class="pyrsm-summary__note">{html.escape(str(note))}</p>')
        parts.append("</section>")
    parts.append("</section>")
    return "".join(parts)
