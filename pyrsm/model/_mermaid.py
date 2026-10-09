"""Display a Mermaid flowchart in a notebook, the way Radiant shows its trees.

Radiant draws decision and classification/regression trees with mermaid (via
DiagrammeR). ``MermaidDiagram`` wraps the mermaid source so that it renders
inline wherever HTML output can run JavaScript (Jupyter, JupyterLab, VS Code,
Colab, Quarto HTML). Front-ends that only render markdown get a ```mermaid
fenced block instead, which Quarto and GitHub draw natively. ``str()`` gives
the raw source, and ``save`` writes a standalone ``.html`` page or the
``.mmd`` source.
"""

from __future__ import annotations

import html
import uuid

MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"

# One loader shared by every diagram on the page, so the library is fetched
# once even when several cells render at the same time.
_LOADER = """
<script>
(function () {
  var el = document.getElementById("%(uid)s");
  if (!el) { return; }
  if (!window.__pyrsmMermaid) {
    window.__pyrsmMermaid = new Promise(function (resolve, reject) {
      if (window.mermaid) { resolve(window.mermaid); return; }
      var s = document.createElement("script");
      s.src = "%(cdn)s";
      s.onload = function () { resolve(window.mermaid); };
      s.onerror = reject;
      document.head.appendChild(s);
    });
  }
  window.__pyrsmMermaid.then(function (mermaid) {
    // tooltips use "click <id> callback" as in Radiant; the callback is a no-op
    window.callback = window.callback || function () {};
    mermaid.initialize({ startOnLoad: false, securityLevel: "loose" });
    mermaid.run({ nodes: el.querySelectorAll("pre.mermaid") });
  });
})();
</script>
"""


class MermaidDiagram:
    """Mermaid flowchart source with rich notebook display.

    Parameters
    ----------
    source : str
        Mermaid source, e.g. ``"graph LR\\n  A --- B"``.
    width : str, default "900px"
        Maximum width of the rendered diagram (any CSS length).
    """

    def __init__(self, source: str, width: str = "900px") -> None:
        self.source = source
        self.width = width

    def __str__(self) -> str:
        return self.source

    def __repr__(self) -> str:
        return self.source

    def _html_fragment(self) -> str:
        uid = f"pyrsm-mermaid-{uuid.uuid4().hex}"
        body = (
            f'<div id="{uid}" style="max-width:{self.width};overflow-x:auto">'
            f'<pre class="mermaid" style="background:none;border:none;margin:0">'
            f"{html.escape(self.source)}</pre></div>"
        )
        return body + _LOADER % {"uid": uid, "cdn": MERMAID_CDN}

    def _repr_html_(self) -> str:
        return self._html_fragment()

    def _repr_markdown_(self) -> str:
        return f"```mermaid\n{self.source}\n```"

    def save(self, path: str) -> None:
        """Save as a standalone HTML page (``.html``) or as mermaid source."""
        if str(path).lower().endswith((".html", ".htm")):
            page = (
                "<!doctype html>\n<html><head><meta charset='utf-8'>"
                "<title>Tree</title></head><body>\n"
                f"{self._html_fragment()}\n</body></html>\n"
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(page)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.source + "\n")
