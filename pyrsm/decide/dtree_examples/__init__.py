"""Decision analysis examples from the Radiant tutorial series, for Python.

The questions, trees and expected answers of the decision-tree videos in the
Radiant Tutorial Series (https://www.youtube.com/playlist?list=PLNhtaetb48EdKRIY7MewCyvb_1x7dV3xw),
recreated for ``pyrsm.decide.dtree``. Each example is a set of named trees:
trees that reference each other (``dtree("name")`` or ``variables: name``)
are in the same set under the names they use.

>>> import pyrsm as rsm
>>> ex = rsm.decide.dtree_example("subtrees")
>>> tree = rsm.decide.dtree(ex["trees"]["main_tree"], subtree_library=ex["trees"])
>>> round(tree.payoff)
955000
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_HERE = Path(__file__).parent

QUESTIONS = {
    1: (
        "You are the owner of Radycal Kicks and deciding whether to open a new store. "
        "You've learned that there is a location available in a trendy district, but "
        "because of high rents it is a risky proposition. The fixed cost of opening the "
        "store is $2 million. Based on historical performance of your business and this "
        "district, you believe there is a 60% chance of the store being successful and "
        "earning $5 million going forward, and a 40% chance of it being a failure and "
        "losing $1 million going forward (these earnings and losses are in addition to "
        "the initial fixed cost of opening the store). What is the optimal strategy, and "
        "what is the Expected Monetary Value (EMV) associated with your strategy?"
    ),
    2: (
        "You are an actor and would like to sign a contract with either a movie company "
        "or a TV network. If you sign with a movie company, you need to pay $5,000 for "
        "the legal fees. There will be a 30% chance of having small box office and "
        "earning $200,000, a 60% chance of having medium box office with payoff equal to "
        "$1,000,000, and a 10% chance of having large box office and getting $3,000,000. "
        "If you decide to sign the contract with TV networks instead, your payoff will be "
        "$900,000. What is EMV of signing the contract with movie company? What is your "
        "optimal strategy?"
    ),
    3: (
        "You are the owner of Radycal Kicks and deciding whether to open a new store. You "
        "believe there is a 60% chance of the store being successful. Now you are offered "
        "to run a short-term popup store to better evaluate store performance in the "
        "district. Running a popup store helps identify whether or not the location will "
        "be successful. You believe that the popup test will correctly identify "
        "successful locations 80% of the time and correctly identify failures 90% of the "
        "time. Given the result of popup test is successful, what is the probability of a "
        "real store being successful?"
    ),
    4: (
        "Continued from Example 1. Now the owner of the retail space offers you the "
        "ability to run a short-term popup store to better evaluate store performance in "
        "the district. Doing so will cost an additional $500,000 but allow you to acquire "
        "potentially helpful information. Running a popup store helps identify whether or "
        "not the location will be successful. You believe that the popup test will "
        "correctly identify successful locations 80% of the time and correctly identify "
        "failures 90% of the time. What is the optimal strategy? What is the EMV "
        "associated with your strategy?"
    ),
}

#: id -> description; ``trees`` maps the tree name used in the workspace (and
#: in ``dtree("name")`` references) to its file in this folder.
EXAMPLES: list[dict[str, Any]] = [
    {
        "id": "open-store",
        "title": "Open a new store (Example 1)",
        "video": "Using Radiant to Construct a Decision Tree (#2)",
        "question": 1,
        "trees": {"open store": "open_store.yaml"},
        "main": "open store",
        "answer": "Open the store: EMV $600,000 (vs. $0 for not opening).",
    },
    {
        "id": "sign-contract",
        "title": "Sign a contract (Example 2)",
        "video": "How to Write Decision Tree Results into a Report (#3)",
        "question": 2,
        "trees": {
            "sign contract": "sign_contract.yaml",
            "fees 10000": "sign_contract_fees_10000.yaml",
            "medium 800000": "sign_contract_medium_800000.yaml",
        },
        "main": "sign contract",
        "answer": (
            "Sign with the movie company: EMV $955,000 vs. $900,000 for the TV network. "
            "With $10,000 in legal fees the EMV is $950,000; with a $800,000 payoff for a "
            "medium box office it drops to $835,000 and signing with the TV network is best."
        ),
    },
    {
        "id": "sensitivity",
        "title": "Sensitivity analysis (Example 2)",
        "video": "Sensitivity Analysis of Decision Tree (#4)",
        "question": 2,
        "trees": {
            "fees 5000": "sign_contract_fees_5000.yaml",
            "fees 8000": "sign_contract_fees_8000.yaml",
            "fees variable": "sign_contract.yaml",
        },
        "main": "fees variable",
        "sensitivity": {
            "tree": "fees variable",
            "vars": {"legal fees": {"min": 0, "max": 100000, "step": 10000}},
            "decisions": ["Sign with Movie Company", "Sign with TV Network"],
        },
        "answer": (
            "Method 1 (edit the value): EMV $955,000 with $5,000 and $952,000 with $8,000 in "
            "legal fees. Method 2 (a variable): below $60,000 in legal fees sign with the "
            "movie company, above $60,000 with the TV network, indifferent at $60,000."
        ),
    },
    {
        "id": "debug",
        "title": "Debug decision tree input (Example 2)",
        "video": "How to Debug Decision Tree Input (#5)",
        "question": 2,
        "trees": {"sign contract with errors": "sign_contract_with_errors.yaml"},
        "main": "sign contract with errors",
        "answer": (
            "Four problems: line 4 'type: chance' is not indented under its node name, "
            "line 9 'p: 0.6' is not indented under 'Medium Box Office:', line 12 has no "
            "probability, and line 13 'payoff' has no ':' and no value. The corrected tree "
            "is the 'sign contract' tree of the Sign a contract example."
        ),
    },
    {
        "id": "imperfect-information",
        "title": "Imperfect information: a popup store (Examples 3 and 4)",
        "video": "Decision Trees with Imperfect Information (#6, #7)",
        "question": 4,
        "trees": {
            "full": "popup_full.yaml",
            "parts": "popup_parts.yaml",
            "popup": "popup.yaml",
        },
        "main": "full",
        "answer": (
            "P(success | positive test) = 0.8 x 0.6 / (0.8 x 0.6 + 0.1 x 0.4) = 0.923. "
            "Run the popup test and open only after a positive result: EMV $820,000 vs. "
            "$600,000 for opening right away. 'parts' gives the same answer by referencing "
            "the 'popup' tree, which uses the variables of 'parts'."
        ),
    },
    {
        "id": "subtrees",
        "title": "Sub-trees (Example 2)",
        "video": "Building a Decision Tree with Sub-trees in Radiant (#8)",
        "question": 2,
        "trees": {"main_tree": "main_tree.yaml", "sub_tree": "sub_tree.yaml"},
        "main": "main_tree",
        "answer": (
            "The sub-tree (signing with the movie company) is worth $955,000; it uses the "
            "legal fees defined in 'main_tree'. Sign with the movie company: EMV $955,000 "
            "vs. $900,000."
        ),
    },
]


def _load(example: dict[str, Any]) -> dict[str, Any]:
    out = dict(example)
    out["trees"] = {
        name: (_HERE / file).read_text(encoding="utf-8") for name, file in example["trees"].items()
    }
    if isinstance(example.get("question"), int):
        out["question_text"] = QUESTIONS[example["question"]]
    return out


def dtree_examples() -> list[dict[str, Any]]:
    """Every example: id, title, video, question, trees (name -> YAML), main tree, answer."""
    return [_load(e) for e in EXAMPLES]


def dtree_example(example_id: str) -> dict[str, Any]:
    """One example by id (see ``dtree_examples()``)."""
    for e in EXAMPLES:
        if e["id"] == example_id:
            return _load(e)
    raise ValueError(f"No decision tree example '{example_id}'. Available: {[e['id'] for e in EXAMPLES]}")
