#!/usr/bin/env python3
"""Generate the decision-tree tutorial notebooks (examples/decide/dtree-tutorial).

Worked decision-analysis examples: the questions, the trees and the
solutions, using ``pyrsm.decide.dtree``.
Trees come from ``pyrsm/decide/dtree_examples`` so the notebooks, the tests
and Radiant's "Load example" menu use the same text.

Writes executed ``.ipynb`` notebooks (cells run in-process, outputs stored)
and matching ``.qmd`` files.

    python scripts/generate_dtree_tutorial.py
"""

from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "decide" / "dtree-tutorial"
QMD = ROOT / "examples" / "qmd" / "decide" / "dtree-tutorial"

SETUP = """import pyrsm as rsm
from IPython.display import Markdown


def show(tree, final=True):
    \"\"\"Draw a decision tree (Mermaid; rendered by JupyterLab, VS Code, Quarto).\"\"\"
    return Markdown(f"```mermaid\\n{tree.to_mermaid(final=final)}\\n```")"""


def tree_cell(var: str, example: str, name: str) -> str:
    import pyrsm as rsm

    text = rsm.decide.dtree_example(example)["trees"][name].rstrip()
    # The YAML starts right after the quotes so line numbers in error
    # messages are the ones in Radiant's editor.
    return f'{var} = """{text}\n"""'


def _questions() -> dict[int, str]:
    import importlib

    return importlib.import_module("pyrsm.decide.dtree_examples").QUESTIONS


def question(n: int) -> str:
    return "> " + _questions()[n]


NOTEBOOKS: list[tuple[str, str, list[tuple[str, str]]]] = []


def nb(name: str, title: str, cells: list[tuple[str, str]]) -> None:
    NOTEBOOKS.append((name, title, cells))


# ---------------------------------------------------------------------------
nb(
    "01-construct-a-tree",
    "Construct a decision tree",
    [
        ("md", "In Radiant use _Model > Decide > Decision analysis_ and pick **Load example > "
         "Open a new store**."),
        ("md", "## Example 1\n\n" + question(1)),
        ("code", SETUP),
        ("md", "Rules for decision-tree input (as in Radiant):\n\n"
         "1. Start with a name (`name: open a new store`).\n"
         "2. The next line starts a `variables:` section or a node (`type: decision` or `type: chance`).\n"
         "3. Every line has a `:`. Node names end with `:`; other lines assign a value "
         "(`payoff: 5000000`, `p: 0.6`, `cost: 2000000`).\n"
         "4. A node type is followed on the next line by a node name.\n"
         "5. Indent (tab or four spaces) the lines after a node name.\n"
         "6. End nodes need a `payoff:`; end nodes of a chance node also need a probability `p:`.\n\n"
         "The $2 million to open the store is a `cost` of the `open` branch: it is subtracted from "
         "the expected payoff of that branch."),
        ("code", tree_cell("open_store", "open-store", "open store")),
        ("code", 'tree = rsm.decide.dtree(open_store)\ntree.summary(input=False, output=True)'),
        ("md", "The final tree folds back from right to left. Opening is worth "
         "0.6 × 5,000,000 + 0.4 × (−1,000,000) − 2,000,000 = **600,000**, and not opening 0, "
         "so the optimal strategy is to **open the store** with an EMV of **$600,000**."),
        ("code", "show(tree)"),
    ],
)

nb(
    "02-trees-in-a-report",
    "Decision trees in a report",
    [
        ("md", "In Radiant: **Load example > Sign a contract** loads the three trees used here; each tree has its own "
         "name, and the report icon adds its Python code to the report."),
        ("md", "## Example 2\n\n" + question(2)),
        ("code", SETUP),
        ("md", "A `variables:` section gives a value a name that the tree can use, here the legal fees. "
         "The fees are a `cost` of signing with the movie company."),
        ("code", tree_cell("sign_contract", "sign-contract", "sign contract")),
        ("code", "tree1 = rsm.decide.dtree(sign_contract)\ntree1.summary(input=False, output=True)"),
        ("md", "The EMV of signing with the movie company is "
         "0.3 × 200,000 + 0.6 × 1,000,000 + 0.1 × 3,000,000 − 5,000 = **955,000**, more than the "
         "900,000 from the TV network: **sign with the movie company**."),
        ("code", "show(tree1)"),
        ("md", "### What if the legal fees are $10,000?"),
        ("code", tree_cell("fees_10000", "sign-contract", "fees 10000")),
        ("code", "tree2 = rsm.decide.dtree(fees_10000)\ntree2.summary(input=False, output=True)"),
        ("md", "The EMV drops to $950,000: still sign with the movie company."),
        ("md", "### What if a medium box office pays $800,000?"),
        ("code", tree_cell("medium_800000", "sign-contract", "medium 800000")),
        ("code", "tree3 = rsm.decide.dtree(medium_800000)\ntree3.summary(input=False, output=True)"),
        ("md", "The movie company is now worth 0.3 × 200,000 + 0.6 × 800,000 + 0.1 × 3,000,000 − "
         "5,000 = 835,000, less than $900,000: **sign with the TV network**."),
        ("code", "show(tree3)"),
    ],
)

nb(
    "03-sensitivity",
    "Sensitivity analysis",
    [
        ("md", "In Radiant: **Load example > Sign a contract: sensitivity analysis**, then the _Sensitivity_ tab."),
        ("md", "## Example 2\n\n" + question(2)),
        ("code", SETUP),
        ("md", "## Method 1: edit the value\n\n### Legal fees of $5,000"),
        ("code", tree_cell("fees_5000", "sensitivity", "fees 5000")),
        ("code", "tree = rsm.decide.dtree(fees_5000)\ntree.summary(input=False, output=True)"),
        ("md", "Signing with the movie company is worth $955,000, the TV network $900,000: sign with "
         "the movie company.\n\n### Legal fees of $8,000"),
        ("code", tree_cell("fees_8000", "sensitivity", "fees 8000")),
        ("code", "tree = rsm.decide.dtree(fees_8000)\ntree.summary(input=False, output=True)"),
        ("md", "With $8,000 in legal fees the movie company is worth $952,000: still the best option."),
        ("md", "## Method 2: use a variable\n\nPut the legal fees in a `variables:` section, then let "
         "them vary from 0 to $100,000 in steps of $10,000."),
        ("code", tree_cell("fees_variable", "sensitivity", "fees variable")),
        ("code", 'tree = rsm.decide.dtree(fees_variable)\n'
                 'sens = tree.sensitivity({"legal fees": {"min": 0, "max": 100000, "step": 10000}})\n'
                 'sens.to_frame()'),
        ("code", 'sens.plot(decisions=["Sign with Movie Company", "Sign with TV Network"])'),
        ("md", "Below $60,000 in legal fees sign with the movie company; above $60,000 sign with the TV "
         "network. At $60,000 the actor is indifferent between the two."),
        ("code", "sens.threshold_flips()"),
    ],
)

nb(
    "04-debug-tree-input",
    "Debug decision tree input",
    [
        ("md", "In Radiant: **Load example > Sign a contract: debug the input** and press _Calculate_."),
        ("code", SETUP),
        ("md", "The tree below has the four most common problems: a missing colon, a missing value, "
         "and two indentation mistakes."),
        ("code", tree_cell("with_errors", "debug", "sign contract with errors")),
        ("code", "tree = rsm.decide.dtree(with_errors)\ntree.summary(input=False)"),
        ("md", "The messages name the lines to fix, as in Radiant:\n\n"
         "- **Line 13** (`payoff`) has no `:` and no value.\n"
         "- **Line 4** (`type: chance`) must be indented under `Sign with Movie Company:` (line 3), "
         "and line 5 must be indented the same as line 4.\n"
         "- **Line 9** (`p: 0.6`) must be indented under `Medium Box Office:`.\n"
         "- **Line 12** has a `p:` without a probability.\n\n"
         "The corrected tree:"),
        ("code", tree_cell("fixed", "sign-contract", "sign contract")),
        ("code", "tree = rsm.decide.dtree(fixed)\ntree.summary(input=False, output=True)"),
    ],
)

nb(
    "05-imperfect-information",
    "Decision trees with imperfect information",
    [
        ("md", "In Radiant: **Load example > Imperfect information: a popup store**."),
        ("md", "## Example 3\n\n" + question(3)),
        ("code", SETUP),
        ("md", "Bayes' rule: P(S|+) = P(+|S) P(S) / P(+), with P(+) = P(+|S) P(S) + P(+|F) P(F)."),
        ("code", "p_s, p_pos_s, p_neg_f = 0.6, 0.8, 0.9\n"
                 "p_pos = p_pos_s * p_s + (1 - p_neg_f) * (1 - p_s)\n"
                 "p_s_pos = p_pos_s * p_s / p_pos\n"
                 "round(p_pos, 4), round(p_s_pos, 4)"),
        ("md", "A positive popup test raises the probability of success from 60% to **92.3%**.\n\n"
         "## Example 4\n\n" + question(4)),
        ("md", "The same calculations as formulas in the `variables:` section: a formula can use any "
         "variable defined above it, and the tree uses the results."),
        ("code", tree_cell("full", "imperfect-information", "full")),
        ("code", "tree = rsm.decide.dtree(full)\ntree.summary(input=False, output=True)"),
        ("md", "Opening right away is worth $600,000 (Example 1). Running the popup test first and "
         "opening only after a positive result is worth **$820,000**: run the popup test."),
        ("code", "show(tree)"),
    ],
)

nb(
    "06-subtrees",
    "Combine decision trees: dtree() references",
    [
        ("md", "The `dtree(\"name\")` way to combine trees, which also works in Radiant for R "
         "(notebook 07 has the newer `tree:` branches). In Radiant: **Load example > Sign a contract: dtree() references** (and **Imperfect information: a popup store** for the "
         "second example). Each tree has a name in the tree list; a tree uses another with "
         "`dtree(\"name\")` in its `variables:` section."),
        ("md", "Two ways to combine trees, as in Radiant:\n\n"
         "- `sign_mc: dtree('sub_tree')` evaluates the tree named `sub_tree` and uses its EMV. The "
         "sub-tree is evaluated with *this* tree's variables, so it uses the legal fees defined here.\n"
         "- `variables: main_tree` (a tree name instead of a list) gives a tree the variables of "
         "`main_tree` (except its `dtree()` references).\n\n"
         "In Python the trees a tree refers to are passed as `subtree_library`: a dict of name → "
         "tree YAML (or `dtree` objects)."),
        ("md", "## Example 2 with a sub-tree\n\n" + question(2)),
        ("code", SETUP),
        ("code", tree_cell("main_tree", "subtrees", "main_tree")),
        ("code", tree_cell("sub_tree", "subtrees", "sub_tree")),
        ("code", 'trees = {"main_tree": main_tree, "sub_tree": sub_tree}\n'
                 'tree = rsm.decide.dtree(main_tree, subtree_library=trees)\n'
                 'tree.summary(input=False, output=True)'),
        ("code", "tree.subtree_refs"),
        ("md", "The sub-tree (signing with the movie company) is worth $955,000, more than the "
         "$900,000 from the TV network. Because the sub-tree uses the main tree's legal fees, a "
         "sensitivity analysis on the main tree flows through to it:"),
        ("code", 'sens = tree.sensitivity({"legal fees": {"min": 0, "max": 100000, "step": 10000}})\n'
                 'sens.plot()'),
        ("md", "## Example 4 in parts\n\nThe popup test as a separate tree (`popup`) that uses the "
         "variables of the main tree (`parts`); `parts` uses the EMV of `popup` as the payoff of "
         "running the test."),
        ("code", tree_cell("parts", "imperfect-information", "parts")),
        ("code", tree_cell("popup", "imperfect-information", "popup")),
        ("code", 'trees = {"parts": parts, "popup": popup}\n'
                 'tree = rsm.decide.dtree(parts, subtree_library=trees)\n'
                 'tree.summary(input=False, output=True)'),
        ("md", "Same answer as the single tree in the previous notebook: run the popup test, EMV "
         "**$820,000**. Splitting a large tree into parts keeps each part readable; move the "
         "variables into the main tree once the parts work."),
    ],
)


nb(
    "07-product-launch",
    "Product launch: sub-trees and a tornado chart",
    [
        ("md", "In Radiant: **Load example > Product launch**. Three trees: the main tree "
         "`product launch` with every variable, and two sub-trees, `launch` (what happens after "
         "a launch) and `test market` (a test market, then a launch decision)."),
        ("md", "## Example 5\n\n" + question(5)),
        ("code", SETUP),
        ("md", "A branch can *be* another tree: `tree: launch` puts the tree `launch` at that "
         "branch. The branch keeps its own label, `p:` and `cost:`, and the combined tree is "
         "solved, printed and plotted as one. The sub-trees take their variables from the main "
         "tree (`variables: product launch`) so each part can also be calculated on its own."),
        ("code", tree_cell("product_launch", "product-launch", "product launch")),
        ("code", tree_cell("launch", "product-launch", "launch")),
        ("md", "After a test market the probability of high demand is no longer 0.4. "
         "`with:` sets values for one use of a tree: after a positive test `launch` is "
         "evaluated with `P(high)` equal to `P(high|pos)` (Bayes' rule, in the main tree's "
         "variables)."),
        ("code", tree_cell("test_market", "product-launch", "test market")),
        ("code", 'trees = {"product launch": product_launch, "launch": launch, '
                 '"test market": test_market}\n'
                 'tree = rsm.decide.dtree(product_launch, subtree_library=trees)\n'
                 'tree.summary(input=False, output=True)'),
        ("md", "Test the market first: EMV **$1,380,000**, against $600,000 for launching right "
         "away. Launch only after a positive test. `tree_uses` lists each use of a sub-tree and "
         "the EMV it contributes:"),
        ("code", "import pandas as pd\n"
                 "pd.DataFrame(tree.tree_uses)"),
        ("md", "The plot draws each sub-tree in its own box. In JupyterLab, VS Code and Radiant, "
         "hovering over a node shows how its value was calculated."),
        ("code", "show(tree)"),
        ("md", "## How sure are we? One variable\n\nHow does the best strategy change with the "
         "probability of high demand? Below 0.2 do not launch, above 0.6 launch right away, and "
         "in between test the market first."),
        ("code", 'sens = tree.sensitivity({"P(high)": {"min": 0.1, "max": 0.7, "step": 0.05}})\n'
                 'sens.plot(["Launch now", "Test market first", "Do not launch"])'),
        ("md", "## Several variables at once: a tornado chart\n\nSet each variable to a low and a "
         "high value, one at a time, with the others at their base values. The bars show how far "
         "the EMV moves; the longest bar is the input that matters most."),
        ("code", 'tor = tree.tornado({\n'
                 '    "P(high)": (0.3, 0.5),\n'
                 '    "margin": (40, 60),\n'
                 '    "fixed cost": (3_500_000, 4_500_000),\n'
                 '    "P(pos|high)": (0.75, 0.95),\n'
                 '    "test cost": (200_000, 400_000),\n'
                 '})\n'
                 'tor.to_frame()'),
        ("code", "tor.plot()"),
        ("md", "The margin and the probability of high demand move the EMV the most, but within "
         "these ranges the decision stays the same: test the market first. Getting a better "
         "estimate of the margin is worth more than haggling over the cost of the test."),
    ],
)


# ---------------------------------------------------------------------------


def _run_cell(code: str, ns: dict) -> list:
    """Run one cell like Jupyter: stdout, then the value of a final expression."""
    outputs = []
    tree = ast.parse(code)
    last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    body = ast.Module(body=tree.body[:-1] if last else tree.body, type_ignores=[])
    # Register the cell's source as Jupyter does, so inspect.getsource works
    # for functions defined in it (pyrsm.decide.simulate reads them)
    import linecache

    filename = f"<notebook-cell-{abs(hash(code))}>"
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(body, filename, "exec"), ns)
        value = eval(compile(ast.Expression(last.value), filename, "eval"), ns) if last else None
    if buf.getvalue():
        outputs.append(nbformat.v4.new_output("stream", name="stdout", text=buf.getvalue()))
    if value is not None:
        outputs.append(_display(value))
    return outputs


def _display(value) -> object:
    from IPython.display import Markdown

    if isinstance(value, Markdown):
        data = {"text/markdown": value.data, "text/plain": "<IPython.core.display.Markdown object>"}
    elif hasattr(value, "save") and hasattr(value, "draw"):  # plotnine
        png = io.BytesIO()
        value.save(png, format="png", dpi=96, verbose=False)
        data = {"image/png": base64.b64encode(png.getvalue()).decode(), "text/plain": "<Figure>"}
    elif hasattr(value, "to_html"):
        data = {"text/html": value.to_html(), "text/plain": str(value)}
    else:
        data = {"text/plain": repr(value)}
    return nbformat.v4.new_output("execute_result", data=data, execution_count=None)


def build(name: str, title: str, cells: list[tuple[str, str]]) -> None:
    book = nbformat.v4.new_notebook()
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    book.cells.append(nbformat.v4.new_markdown_cell(f"# {title}"))
    ns: dict = {}
    count = 0
    qmd = [f"---\ntitle: \"{title}\"\njupyter: python3\n---\n"]
    for kind, text in cells:
        if kind == "md":
            book.cells.append(nbformat.v4.new_markdown_cell(text))
            qmd.append(text + "\n")
        else:
            count += 1
            cell = nbformat.v4.new_code_cell(text, execution_count=count)
            cell.outputs = _run_cell(text, ns)
            for out in cell.outputs:
                if out.output_type == "execute_result":
                    out.execution_count = count
            book.cells.append(cell)
            qmd.append(f"```{{python}}\n{text}\n```\n")
    nbformat.validate(book)
    (OUT / f"{name}.ipynb").write_text(nbformat.writes(book) + "\n", encoding="utf-8")
    (QMD / f"{name}.qmd").write_text("\n".join(qmd), encoding="utf-8")


def readme() -> None:
    import pyrsm as rsm

    lines = [
        "# Decision analysis tutorial (Python)",
        "",
        "Worked decision-analysis examples using `pyrsm.decide.dtree`. The trees are also "
        "available in Radiant under _Model > Decide > Decision analysis > Load example_ and in "
        "Python with `rsm.decide.dtree_examples()`.",
        "",
        "| Notebook | Topic | Example |",
        "|---|---|---|",
        "| [01-construct-a-tree](01-construct-a-tree.ipynb) | Enter and solve a tree | 1 |",
        "| [02-trees-in-a-report](02-trees-in-a-report.ipynb) | Several trees, results in a report | 2 |",
        "| [03-sensitivity](03-sensitivity.ipynb) | Sensitivity analysis | 2 |",
        "| [04-debug-tree-input](04-debug-tree-input.ipynb) | Fix input errors | 2 |",
        "| [05-imperfect-information](05-imperfect-information.ipynb) | Imperfect information, Bayes' rule | 3, 4 |",
        "| [06-subtrees](06-subtrees.ipynb) | Combine trees with `dtree()` | 2, 4 |",
        "| [07-product-launch](07-product-launch.ipynb) | Sub-trees with `tree:` and `with:`, tornado chart | 5 |",
        "",
        "## Questions",
        "",
    ]
    for n, text in _questions().items():
        lines += [f"**Example {n}.** {text}", ""]
    lines += ["## Answers", ""]
    for ex in rsm.decide.dtree_examples():
        lines += [f"- **{ex['title']}**: {ex['answer']}"]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    QMD.mkdir(parents=True, exist_ok=True)
    for name, title, cells in NOTEBOOKS:
        build(name, title, cells)
        print("wrote", name)
    readme()
    print(json.dumps({"notebooks": len(NOTEBOOKS), "out": str(OUT)}))


if __name__ == "__main__":
    main()
