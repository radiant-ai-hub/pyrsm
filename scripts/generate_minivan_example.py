#!/usr/bin/env python
"""Generate the minivan choice-based-conjoint (CBC) example datasets.

The study design follows the minivan example in Chapman and Feit, *R for
Marketing Research and Analytics* (Springer, chapter 13): 200 respondents
each answer 15 choice tasks with 3 minivans, described by seating (6, 7, 8),
cargo space (2ft, 3ft), engine (gas, hybrid, electric) and price ($30K, $35K,
$40K); about 30% of the respondents carpool. The data are simulated here from
known preferences (not the book's file), with price numeric (in $1,000s) so
willingness to pay follows from a single price coefficient:

* every respondent has their own part-worths, drawn around the averages in
  ``MEAN`` with spread ``SD`` (preferences differ between people);
* carpoolers like extra seats: +1.5 for 7 seats and +2.0 for 8 seats;
* each choice task shows 3 different profiles drawn at random from the 54
  possible ones, and the respondent picks the profile with the highest
  utility plus logit (Gumbel) noise.

Also writes ``minivan_market``: six minivans competing in one market (the
designs in the book's share simulation) for Predict > Dataset.

Deterministic (fixed seed). Usage: ``python scripts/generate_minivan_example.py``
"""

from itertools import product
from pathlib import Path

import numpy as np
import polars as pl

DIR = Path(__file__).resolve().parents[1] / "examples" / "data" / "multivariate"

SEAT = ["6", "7", "8"]
CARGO = ["2ft", "3ft"]
ENG = ["gas", "hyb", "elec"]
PRICE = [30, 35, 40]

#: average part-worths (first level of each attribute = 0); price per $1,000
MEAN = {"seat7": -1.0, "seat8": -1.0, "cargo3ft": 0.5, "enghyb": -1.0,
        "engelec": -2.0, "price": -0.2}
#: spread of the part-worths across respondents
SD = {"seat7": 0.5, "seat8": 1.0, "cargo3ft": 0.3, "enghyb": 0.5,
      "engelec": 1.0, "price": 0.05}
CARPOOL_SHIFT = {"seat7": 1.5, "seat8": 2.0}

N_RESP, N_QUES, N_ALT = 200, 15, 3


def utility(beta: dict, seat: str, cargo: str, eng: str, price: int) -> float:
    return (
        (beta["seat7"] if seat == "7" else 0.0)
        + (beta["seat8"] if seat == "8" else 0.0)
        + (beta["cargo3ft"] if cargo == "3ft" else 0.0)
        + (beta["enghyb"] if eng == "hyb" else 0.0)
        + (beta["engelec"] if eng == "elec" else 0.0)
        + beta["price"] * price
    )


def main() -> None:
    rng = np.random.default_rng(1234)
    profiles = list(product(SEAT, CARGO, ENG, PRICE))
    rows = []
    for resp in range(1, N_RESP + 1):
        carpool = "yes" if rng.random() < 0.3 else "no"
        beta = {k: rng.normal(MEAN[k], SD[k]) for k in MEAN}
        if carpool == "yes":
            for k, v in CARPOOL_SHIFT.items():
                beta[k] += v
        for ques in range(1, N_QUES + 1):
            task = (resp - 1) * N_QUES + ques
            shown = rng.choice(len(profiles), size=N_ALT, replace=False)
            util = np.array([utility(beta, *profiles[k]) for k in shown])
            chosen = int(np.argmax(util + rng.gumbel(size=N_ALT)))
            for alt, k in enumerate(shown, start=1):
                seat, cargo, eng, price = profiles[k]
                rows.append(
                    {"resp": resp, "ques": ques, "task": task, "alt": alt,
                     "carpool": carpool, "seat": seat, "cargo": cargo,
                     "eng": eng, "price": price, "choice": int(alt - 1 == chosen)}
                )

    df = pl.DataFrame(rows).with_columns(
        pl.col("carpool").cast(pl.Enum(["yes", "no"])),
        pl.col("seat").cast(pl.Enum(SEAT)),
        pl.col("cargo").cast(pl.Enum(CARGO)),
        pl.col("eng").cast(pl.Enum(ENG)),
        pl.col("price").cast(pl.Int32),
    )
    df.write_parquet(DIR / "minivan.parquet")
    print(f"minivan: {df.height} rows, {df['task'].n_unique()} choice tasks")

    market = pl.DataFrame(
        {
            "product": [1, 2, 3, 4, 5, 6],
            "seat": ["7", "6", "8", "7", "6", "7"],
            "cargo": ["2ft", "2ft", "2ft", "3ft", "2ft", "2ft"],
            "eng": ["hyb", "gas", "gas", "gas", "elec", "hyb"],
            "price": [30, 30, 30, 40, 40, 35],
        }
    ).with_columns(
        pl.col("seat").cast(pl.Enum(SEAT)),
        pl.col("cargo").cast(pl.Enum(CARGO)),
        pl.col("eng").cast(pl.Enum(ENG)),
        pl.col("price").cast(pl.Int32),
    )
    market.write_parquet(DIR / "minivan_market.parquet")
    print(f"minivan_market: {market.height} products")


if __name__ == "__main__":
    main()
