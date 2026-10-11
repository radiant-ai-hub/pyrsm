"""Worked simulation examples for ``pyrsm.decide.simulate``.

Each example has a question, a simulation spec and (where it helps) the
settings of a repeated simulation. Radiant's Model > Simulate "Load example"
menu and the notebooks in ``examples/decide/simulate-tutorial`` use the same
specs.

>>> import pyrsm as rsm
>>> ex = rsm.decide.simulate_example("profit")
>>> sim = rsm.decide.simulate(ex["spec"])
"""

from __future__ import annotations

import copy
from typing import Any

EXAMPLES: list[dict[str, Any]] = [
    {
        "id": "profit",
        "title": "Daily profit (and a year of it)",
        "question": (
            "A company sells a product at $6 (30% of days) or $8 (70% of days). Demand is "
            "1000 - 50 x price plus a normally distributed error with mean 0 and standard "
            "deviation 100. The variable cost is $5 per unit and the fixed cost $1,000 per day. "
            "What is the distribution of daily profit, and how likely is a profit below $100? "
            "How likely is a year (365 days) with a total profit below $36,500?"
        ),
        "spec": {
            "name": "simdat",
            "runs": 1000,
            "seed": 1234,
            "variables": [
                {"name": "var_cost", "kind": "constant", "value": 5},
                {"name": "fixed_cost", "kind": "constant", "value": 1000},
                {"name": "E", "kind": "normal", "mean": 0, "sd": 100},
                {"name": "price", "kind": "discrete", "values": [6, 8], "probs": [0.3, 0.7]},
            ],
            "formulas": [
                "demand = 1000 - 50 * price + E",
                "profit = demand * (price - var_cost) - fixed_cost",
                "profit_small = profit < 100",
            ],
        },
        "repeat": {
            "name": "repdat",
            "reps": 365,
            "vars": ["E", "price"],
            "sum_vars": ["profit"],
            "by": "sim",
            "fun": ["sum"],
            "formulas": ["profit_365 = profit_sum < 36500"],
            "seed": 1234,
        },
        "answer": (
            "Average daily profit is about $500 with a standard deviation of about $560; profit is "
            "below $100 on about 28% of days. Summed over 365 days the annual profit averages about "
            "$172,000 (sd about $10,800) and is never below $36,500."
        ),
    },
    {
        "id": "optimal-price",
        "title": "Optimal price (grid search)",
        "question": (
            "Demand for an ice cream bar is 1000 - 80 x price and each bar costs $2 to make. "
            "Which price between $4 and $10 maximizes profit? Use a grid search over price in "
            "steps of $0.01."
        ),
        "spec": {
            "name": "pricing",
            "runs": 1000,
            "seed": 1234,
            "variables": [
                {"name": "cost", "kind": "constant", "value": 2},
                {"name": "price", "kind": "grid", "min": 4, "max": 10, "step": 0.01},
            ],
            "formulas": [
                "demand = 1000 - 80 * price",
                "profit = demand * (price - cost)",
                "# the price in the row with the highest profit",
                "optimal_price = price[arg_max(profit)]",
                "max_profit = max(profit)",
            ],
        },
        "answer": (
            "Profit is highest at a price of $7.25 (profit $2,205). The grid has 601 prices, "
            "so the simulation has 601 rows whatever the number of simulations."
        ),
    },
    {
        "id": "bank-queue",
        "title": "Bank queue (waiting times)",
        "question": (
            "Customers arrive at a bank on average every 2 minutes (exponential times between "
            "arrivals) and service takes on average 3.5 minutes (exponential). With 2 tellers, "
            "how long do customers wait, how long is the line, and how often does someone wait "
            "more than 5 minutes? How does the average wait change with 1 to 4 tellers?"
        ),
        "spec": {
            "name": "queue",
            "runs": 2000,
            "seed": 1234,
            "variables": [
                {"name": "tellers", "kind": "constant", "value": 2},
                {"name": "gap", "kind": "exponential", "rate": 0.5},
                {"name": "service", "kind": "exponential", "rate": "1 / 3.5"},
            ],
            "formulas": [
                "# minutes each customer waits before a teller is free",
                "wait = queue_wait(gap, service, tellers)",
                "# customers already waiting when a customer arrives",
                "line = queue_length(gap, service, tellers)",
                "in_bank = wait + service",
                "wait_5 = wait > 5",
                "utilization = sum(service) / (tellers * (sum(gap) + in_bank[-1]))",
            ],
        },
        "repeat": {
            "name": "queue_tellers",
            "reps": 4,
            "vars": ["gap", "service"],
            "sum_vars": ["wait", "wait_5"],
            "by": "rep",
            "fun": ["mean", "p95"],
            "grid": [{"name": "tellers", "min": 1, "max": 4, "step": 1}],
            "seed": 1234,
        },
        "answer": (
            "With 2 tellers the tellers are busy about 86% of the time, customers wait about 9.5 "
            "minutes on average and 55% wait more than 5 minutes. Repeating with 1 to 4 tellers: "
            "with one teller the line keeps growing (service is slower than arrivals); with 3 "
            "tellers the average wait drops below a minute, with 4 to about 8 seconds."
        ),
    },
    {
        "id": "gamblers-ruin",
        "title": "Gambler's ruin (custom function)",
        "question": (
            "You start with $1,000 and each day bet 5% of what you have. You win with probability "
            "0.2, and a win pays 5 times the bet; otherwise you lose the bet. What is your wealth "
            "after a year (365 bets)? How likely are you to end with less than you started with?"
        ),
        "spec": {
            "name": "gamble",
            "runs": 365,
            "seed": 1234,
            "variables": [
                {"name": "start", "kind": "constant", "value": 1000},
                {"name": "perc_invest", "kind": "constant", "value": 0.05},
                {"name": "win_mult", "kind": "constant", "value": 5},
                {"name": "win", "kind": "binomial", "trials": 1, "prob": 0.2},
            ],
            "functions": (
                "def calc_wealth(win, start, perc_invest, win_mult):\n"
                "    # wealth at the end of each day: bet a share of what you have\n"
                "    wealth = zeros(len(win))\n"
                "    current = start\n"
                "    for day in range(len(win)):\n"
                "        bet = perc_invest * current\n"
                "        if win[day] == 1:\n"
                "            current = current + win_mult * bet\n"
                "        else:\n"
                "            current = current - bet\n"
                "        wealth[day] = current\n"
                "    return wealth\n"
            ),
            "formulas": [
                "wealth = calc_wealth(win, start, perc_invest, win_mult)",
                "final = wealth[-1]",
                "loss = final < start",
            ],
        },
        "repeat": {
            "name": "gamble_years",
            "reps": 1000,
            "vars": ["win"],
            "sum_vars": ["wealth"],
            "by": "rep",
            "fun": ["last", "min"],
            "formulas": ["loss = wealth_last < 1000", "ruined = wealth_min < 100"],
            "seed": 1234,
        },
        "answer": (
            "Each bet has a positive expected value (0.2 x 5 - 0.8 = +0.2 per dollar bet), yet "
            "over 1,000 simulated years the median final wealth is about $3,700 while the mean is "
            "about $45,000: a few lucky years are worth millions. About 26% of years end below "
            "the $1,000 you started with, and in about 10% wealth drops below $100 at some point."
        ),
    },
    {
        "id": "stock-returns",
        "title": "Correlated stock returns",
        "question": (
            "Simulate 1,000 monthly returns for three stocks with means 1%, 1.5% and 0.8%, "
            "standard deviations 5%, 8% and 3%, and correlations 0.3 (A-B), 0.1 (A-C) and 0.2 "
            "(B-C). Save the result as 'returns' to use it as input data for the portfolio example."
        ),
        "spec": {
            "name": "returns",
            "runs": 1000,
            "seed": 1234,
            "normal_exact": True,
            "variables": [
                {"name": "A", "kind": "normal", "mean": 0.01, "sd": 0.05},
                {"name": "B", "kind": "normal", "mean": 0.015, "sd": 0.08},
                {"name": "C", "kind": "normal", "mean": 0.008, "sd": 0.03},
            ],
            "correlations": [
                {"a": "A", "b": "B", "r": 0.3},
                {"a": "A", "b": "C", "r": 0.1},
                {"a": "B", "b": "C", "r": 0.2},
            ],
        },
        "answer": (
            "With 'exact' checked the simulated returns have exactly the means, standard "
            "deviations and correlations given."
        ),
    },
    {
        "id": "portfolio",
        "title": "Portfolio weights (input data)",
        "question": (
            "Using the simulated 'returns' as input data, which weights for stocks A and B (C gets "
            "the rest) give the lowest risk (standard deviation of the portfolio return)? Search "
            "weights from 0 to 1 in steps of 0.05."
        ),
        "spec": {
            "name": "portfolio",
            "runs": 1000,
            "seed": 1234,
            "data": "returns",
            "variables": [
                {"name": "wA", "kind": "grid", "min": 0, "max": 1, "step": 0.05},
                {"name": "wB", "kind": "grid", "min": 0, "max": 1, "step": 0.05},
            ],
            "formulas": [
                "wC = 1 - wA - wB",
                "valid = wC >= 0",
                "# 99 for weights that do not add up",
                "risk = ifelse(valid, weighted_sum_std(wA, wB, wC, A, B, C), 99)",
                "expected = wA * mean(A) + wB * mean(B) + wC * mean(C)",
                "min_risk_A = wA[arg_min(risk)]",
                "min_risk_B = wB[arg_min(risk)]",
            ],
        },
        "requires": ["stock-returns"],
        "answer": (
            "The lowest-risk portfolio puts most of the money in the least volatile stock (C) "
            "with a small share in A: about 25% A, 0% B and 75% C."
        ),
    },
]


def simulate_examples() -> list[dict[str, Any]]:
    """Every example: id, title, question, spec, repeat settings and answer."""
    return [copy.deepcopy(e) for e in EXAMPLES]


def simulate_example(example_id: str) -> dict[str, Any]:
    """One example by id (see ``simulate_examples()``)."""
    for e in EXAMPLES:
        if e["id"] == example_id:
            return copy.deepcopy(e)
    raise ValueError(
        f"No simulation example '{example_id}'. Available: {[e['id'] for e in EXAMPLES]}"
    )


__all__ = ["EXAMPLES", "simulate_example", "simulate_examples"]
