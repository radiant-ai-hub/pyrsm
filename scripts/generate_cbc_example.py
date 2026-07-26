#!/usr/bin/env python
"""Generate the synthetic choice-based-conjoint (CBC) example dataset.

Simulates a chocolate-bar choice study: 200 respondents x 8 choice tasks x 3
alternatives, with attributes brand / chocolate-type / price / nuts and a known
set of part-worths. Choices are drawn from a logit (Gumbel-noise) model so the
estimated part-worths recover the data-generating values. Deterministic (fixed
seed) so the shipped parquet is reproducible.

Usage:  python scripts/generate_cbc_example.py
"""

from pathlib import Path

import numpy as np
import polars as pl

OUT = Path(__file__).resolve().parents[1] / "examples" / "data" / "multivariate" / "choc.parquet"

BRANDS = ["Hershey", "Lindt", "Godiva"]
TYPES = ["Milk", "Dark", "White"]
PRICES = [2.99, 3.99, 4.99]
NUTS = ["No", "Yes"]

# true part-worths (utilities) used to simulate choices
PW_BRAND = {"Hershey": 0.0, "Lindt": 0.8, "Godiva": 1.3}
PW_TYPE = {"Milk": 0.0, "Dark": 0.5, "White": -0.4}
PW_PRICE = -0.9
PW_NUTS = {"No": 0.0, "Yes": 0.3}


def main() -> None:
    rng = np.random.default_rng(1234)
    rows = []
    choice_id = 0
    for resp in range(1, 201):
        for _task in range(1, 9):
            choice_id += 1
            alts = []
            for alt in range(1, 4):
                br = BRANDS[rng.integers(3)]
                ty = TYPES[rng.integers(3)]
                pr = PRICES[rng.integers(3)]
                nu = NUTS[rng.integers(2)]
                u = PW_BRAND[br] + PW_TYPE[ty] + PW_PRICE * pr + PW_NUTS[nu]
                alts.append((resp, choice_id, alt, br, ty, pr, nu, u))
            util = np.array([a[7] for a in alts]) + rng.gumbel(size=3)
            chosen = int(np.argmax(util))
            for k, a in enumerate(alts):
                rows.append(
                    {
                        "resp": a[0],
                        "choice_id": a[1],
                        "alt": a[2],
                        "brand": a[3],
                        "chocolate": a[4],
                        "price": a[5],
                        "nuts": a[6],
                        "chosen": int(k == chosen),
                    }
                )

    df = pl.DataFrame(rows).with_columns(
        pl.col("brand").cast(pl.Enum(BRANDS)),
        pl.col("chocolate").cast(pl.Enum(TYPES)),
        pl.col("nuts").cast(pl.Enum(NUTS)),
    )
    df.write_parquet(OUT)
    print(f"wrote {OUT}  ({df.height} rows, {df['choice_id'].n_unique()} choice tasks)")


if __name__ == "__main__":
    main()
