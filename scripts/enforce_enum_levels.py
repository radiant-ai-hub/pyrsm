"""Enforce Enum dtypes and a deliberate level order on the example datasets.

Policy: every categorical column in ``examples/data/**.parquet`` is a polars
``Enum`` whose category order is a deliberate, meaningful order, never an
accident of alphabetical sorting.

* Datasets that come from the R radiant packages use the factor level order of
  the R dataset (radiant.data, radiant.basics, radiant.model,
  radiant.multivariate, radiant.design). Those lists are copied verbatim from
  ``levels(<dataset>$<column>)`` in R (trailing whitespace stripped, e.g. R's
  ``toothpaste$gender`` level ``"Male  "``). R wins even where another order
  would look more natural (mp3 ``Size`` is Large, Medium, Small;
  direct_marketing ``History`` puts None last; ketchup ``id`` is R's
  alphabetical C1, C10, C100, ...), so that pyrsm output matches Radiant.
* Columns that are character (not factor) in R, and datasets without an R
  counterpart, get the order documented next to their entry in ``SPEC``.
* Identifiers, free text and join keys stay ``String`` (see ``KEEP_STRING``).

Every String column must be listed in either ``SPEC`` or ``KEEP_STRING``, so a
new dataset with an unclassified text column makes this script fail loudly.

The script rewrites a parquet file only if a dtype or category order changes,
and verifies that no row value changes (same shape, same column order, same
values as strings, same nulls, non-categorical columns bit-for-bit equal).

Usage::

    uv run python scripts/enforce_enum_levels.py           # apply
    uv run python scripts/enforce_enum_levels.py --check   # report only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1] / "examples" / "data"

NO_YES = ["No", "Yes"]
no_yes = ["no", "yes"]
YES_NO = ["Yes", "No"]
yes_no = ["yes", "no"]
FEMALE_MALE = ["Female", "Male"]

# --- shared level lists (from R) ----------------------------------------------
DIAMONDS = {  # radiant.data::diamonds (radiant.model ships the same data)
    "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
    "cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],
    "color": ["D", "E", "F", "G", "H", "I", "J"],
}
TITANIC = {  # radiant.data::titanic
    "pclass": ["1st", "2nd", "3rd"],
    "survived": YES_NO,
    "sex": ["female", "male"],
    "embarked": ["Cherbourg", "Queenstown", "Southampton"],
}
# radiant.model::ketchup$id is R's (alphabetical) factor order: C1, C10, C100, ...
KETCHUP_ID = sorted(f"C{i}" for i in range(1, 301))

# Character in R (avengers/superheroes are dplyr join/bind examples). The two
# tables must share identical Enum dtypes so pl.concat and joins on all columns
# keep working. Order chosen: positive level first, as in radiant's yes/no and
# Yes/No factors; female before male as in every radiant dataset with gender.
HEROES = {
    "alignment": ["good", "bad"],
    "gender": ["female", "male"],
}

# fraud_data (Kaggle, no R counterpart).
# category: merchant types have no natural order -> alphabetical, chosen on
#   purpose so tables read like an index.
# gender: F before M, consistent with Female/Male elsewhere.
# state: US postal codes (+ DC) in alphabetical order, the conventional listing
#   order; only the 50 codes that occur (no DE) so no empty level is added.
FRAUD_CATEGORY = [
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel",
]  # fmt: skip
FRAUD_STATE = [
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "FL", "GA", "HI", "IA",
    "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS",
    "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA",
    "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
]  # fmt: skip

# --- dataset -> column -> levels -------------------------------------------------
SPEC: dict[str, dict[str, list[str]]] = {
    # radiant.basics
    "basics/consider": {"consider": yes_no},
    "basics/newspaper": {
        "Income": ["Low Income", "High Income"],
        "Newspaper": ["WS Journal", "USA Today"],
    },
    "basics/salary": {
        "rank": ["AsstProf", "AssocProf", "Prof"],
        "discipline": ["A", "B"],
        "sex": FEMALE_MALE,
    },
    # radiant.data
    "data/avengers": HEROES,
    "data/superheroes": HEROES,
    "data/diamonds": DIAMONDS,
    "data/titanic": TITANIC,
    # radiant.design
    "design/rndnames": {"Gender": FEMALE_MALE},
    # radiant.model
    "model/diamonds": DIAMONDS,
    "model/titanic": TITANIC,
    "model/direct_marketing": {
        "Age": ["Young", "Middle", "Old"],
        "Gender": FEMALE_MALE,
        "OwnHome": ["Own", "Rent"],
        "Married": ["Married", "Single"],
        "Location": ["Close", "Far"],
        "Children": ["0", "1", "2", "3"],
        "History": ["Low", "Medium", "High", "None"],
    },
    "model/dvd": {"buy": yes_no},
    "model/houseprices": {"brick": NO_YES, "neighborhood": ["East", "North", "West"]},
    "model/ketchup": {
        "id": KETCHUP_ID,
        "choice": ["heinz28", "heinz32", "heinz41", "hunts32"],
        **{f"{p}.{b}": no_yes for p in ("disp", "feat")
           for b in ("heinz28", "heinz32", "heinz41", "hunts32")},
    },  # fmt: skip
    "model/ratings": {
        "Users": [f"U{i}" for i in range(1, 12)],
        "Movies": [f"M{i}" for i in range(1, 11)],
    },
    "model/fraud_data": {
        "category": FRAUD_CATEGORY,
        "gender": ["F", "M"],
        "state": FRAUD_STATE,
    },
    # radiant.multivariate
    "multivariate/carpet": {
        "design": ["A", "B", "C"],
        "brand": ["K2R", "Glory", "Bissell"],
        "price": ["1.19", "1.39", "1.59"],
        "seal": NO_YES,
        "money_back": NO_YES,
    },
    "multivariate/movie": {
        "price": ["$10", "$13", "$16"],
        "sight": ["Staggered", "Not Staggered"],
        "comfort": ["Average no cup", "Average cup", "Large cup"],
        "audio_visual": ["Small plain", "Large plain", "Large digital"],
        "food": ["No food", "Hot dogs and popcorn", "Gourmet food"],
    },
    "multivariate/mp3": {
        "Memory": ["4GB", "6GB", "8GB"],
        "Radio": NO_YES,
        "Size": ["Large", "Medium", "Small"],
        "Price": ["$50", "$100", "$150"],
        "Shape": ["Circular", "Rectangular", "Square"],
    },
    "multivariate/toothpaste": {
        "age.cat": ["< 30", "30 to 39", "> 40"],
        "gender": FEMALE_MALE,
    },
    # Synthetic CBC data (no R counterpart): the order of
    # scripts/generate_cbc_example.py and scripts/generate_conjoint_choice_reference.R,
    # base level first (the level whose true part-worth is 0).
    "multivariate/choc": {
        "brand": ["Hershey", "Lindt", "Godiva"],
        "chocolate": ["Milk", "Dark", "White"],
        "nuts": NO_YES,
    },
}

# Columns that deliberately stay String: identifiers, labels, free text, keys.
KEEP_STRING: dict[str, dict[str, str]] = {
    "data/avengers": {
        "name": "identifier",
        "publisher": "join key (character in R); values differ across the "
        "superheroes/publishers/avengers tables, so Enums would not join",
    },
    "data/superheroes": {"name": "identifier", "publisher": "join key"},
    "data/publishers": {"publisher": "join key, one row per publisher"},
    "data/titanic": {"name": "identifier", "cabin": "free text (174 values)"},
    "model/titanic": {"name": "identifier", "cabin": "free text (174 values)"},
    "design/rndnames": {"Names": "identifier"},
    "model/fraud_data": {
        "trans_date_trans_time": "timestamp stored as text",
        "merchant": "identifier (693 values)",
        "first": "personal name",
        "last": "personal name",
        "street": "address",
        "city": "address (849 values)",
        "job": "free text (478 values)",
        "dob": "date stored as text",
        "trans_num": "transaction identifier",
    },
    "multivariate/city": {"from": "MDS object label", "to": "MDS object label"},
    "multivariate/city2": {"from": "MDS object label", "to": "MDS object label"},
    "multivariate/tpbrands": {"id1": "MDS object label", "id2": "MDS object label"},
    "multivariate/computer": {"brand": "row label for the perceptual map"},
    "multivariate/retailers": {"retailer": "row label for the perceptual map"},
}


def enforce(df: pl.DataFrame, spec: dict[str, list[str]]) -> pl.DataFrame:
    exprs = []
    for col, levels in spec.items():
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in data")
        if len(set(levels)) != len(levels):
            raise ValueError(f"{col}: duplicate levels")
        present = set(df[col].cast(pl.String).drop_nulls().unique().to_list())
        missing = present - set(levels)
        if missing:
            raise ValueError(f"{col}: values not in level list: {sorted(missing)}")
        exprs.append(pl.col(col).cast(pl.String).cast(pl.Enum(levels)))
    return df.with_columns(exprs)


def verify_same_rows(old: pl.DataFrame, new: pl.DataFrame, cols: list[str]) -> None:
    assert old.shape == new.shape, "shape changed"
    assert old.columns == new.columns, "column order changed"
    for c in old.columns:
        if c in cols:
            a, b = old[c].cast(pl.String), new[c].cast(pl.String)
            assert a.equals(b), f"{c}: values changed"
        else:
            assert old[c].dtype == new[c].dtype, f"{c}: dtype changed"
            assert old[c].equals(new[c]), f"{c}: values changed"


def describe(dtype: pl.DataType) -> str:
    if isinstance(dtype, pl.Enum):
        cats = dtype.categories.to_list()
        return f"Enum{cats if len(cats) <= 8 else cats[:4] + ['...']}"
    return str(dtype)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="report, do not write")
    args = ap.parse_args()

    problems, changed = [], []
    files = sorted(ROOT.glob("*/*.parquet"))
    keys = {f"{p.parent.name}/{p.stem}" for p in files}
    for key in sorted(set(SPEC) | set(KEEP_STRING)):
        if key not in keys:
            problems.append(f"{key}: listed in SPEC/KEEP_STRING but no parquet file")

    for path in files:
        key = f"{path.parent.name}/{path.stem}"
        old = pl.read_parquet(path)
        spec = SPEC.get(key, {})
        keep = KEEP_STRING.get(key, {})

        for c, t in old.schema.items():
            if c in spec or c in keep:
                continue
            if t in (pl.String, pl.Categorical) or isinstance(t, pl.Enum):
                problems.append(f"{key}.{c}: {t} column not classified in SPEC/KEEP_STRING")
        for c in keep:
            if c in spec:
                problems.append(f"{key}.{c}: in both SPEC and KEEP_STRING")
            elif old.schema.get(c) != pl.String:
                problems.append(f"{key}.{c}: KEEP_STRING but dtype is {old.schema.get(c)}")

        new = enforce(old, spec)
        verify_same_rows(old, new, list(spec))
        diffs = [
            (c, old.schema[c], new.schema[c])
            for c in spec
            if old.schema[c] != new.schema[c]
        ]
        if not diffs:
            continue
        changed.append(key)
        for c, a, b in diffs:
            print(f"{key}.{c}: {describe(a)} -> {describe(b)}")
        if not args.check:
            new.write_parquet(path)
            reread = pl.read_parquet(path)
            assert reread.schema == new.schema, f"{key}: schema did not round-trip"
            verify_same_rows(old, reread, list(spec))

    for p in problems:
        print("PROBLEM:", p, file=sys.stderr)
    verb = "would change" if args.check else "rewrote"
    print(f"{verb} {len(changed)} file(s): {', '.join(changed) or '-'}")
    return 1 if problems or (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
