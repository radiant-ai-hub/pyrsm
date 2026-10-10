# Simulation tutorial (Python)

Worked simulation examples with `pyrsm.decide.simulate` and `pyrsm.decide.repeat_simulate`. The same examples are in Radiant under _Model > Decide > Simulate > Load example_ and in Python with `rsm.decide.simulate_examples()`.

| Notebook | Topic |
|---|---|
| [01-profit](01-profit.ipynb) | Constants, normal and discrete variables, formulas, repeated simulation (group by Simulation) |
| [02-optimal-price](02-optimal-price.ipynb) | Grid search and `arg_max` |
| [03-bank-queue](03-bank-queue.ipynb) | Exponential arrivals, `queue_wait`, repeat with a grid over a constant |
| [04-gamblers-ruin](04-gamblers-ruin.ipynb) | A custom function with a loop; repeat with `last` and `min` |
| [05-portfolio](05-portfolio.ipynb) | Correlated normals, input data, `weighted_sum_std` |

## Questions and answers

**Daily profit (and a year of it).** A company sells a product at $6 (30% of days) or $8 (70% of days). Demand is 1000 - 50 x price plus a normally distributed error with mean 0 and standard deviation 100. The variable cost is $5 per unit and the fixed cost $1,000 per day. What is the distribution of daily profit, and how likely is a profit below $100? How likely is a year (365 days) with a total profit below $36,500?

_Answer:_ Average daily profit is about $500 with a standard deviation of about $560; profit is below $100 on about 28% of days. Summed over 365 days the annual profit averages about $172,000 (sd about $10,800) and is never below $36,500.

**Optimal price (grid search).** Demand for an ice cream bar is 1000 - 80 x price and each bar costs $2 to make. Which price between $4 and $10 maximizes profit? Use a grid search over price in steps of $0.01.

_Answer:_ Profit is highest at a price of $7.25 (profit $2,205). The grid has 601 prices, so the simulation has 601 rows whatever the number of simulations.

**Bank queue (waiting times).** Customers arrive at a bank on average every 2 minutes (exponential times between arrivals) and service takes on average 3.5 minutes (exponential). With 2 tellers, how long do customers wait, how long is the line, and how often does someone wait more than 5 minutes? How does the average wait change with 1 to 4 tellers?

_Answer:_ With 2 tellers the tellers are busy about 86% of the time, customers wait about 9.5 minutes on average and 55% wait more than 5 minutes. Repeating with 1 to 4 tellers: with one teller the line keeps growing (service is slower than arrivals); with 3 tellers the average wait drops below a minute, with 4 to about 8 seconds.

**Gambler's ruin (custom function).** You start with $1,000 and each day bet 5% of what you have. You win with probability 0.2, and a win pays 5 times the bet; otherwise you lose the bet. What is your wealth after a year (365 bets)? How likely are you to end with less than you started with?

_Answer:_ Each bet has a positive expected value (0.2 x 5 - 0.8 = +0.2 per dollar bet), yet over 1,000 simulated years the median final wealth is about $3,700 while the mean is about $45,000: a few lucky years are worth millions. About 26% of years end below the $1,000 you started with, and in about 10% wealth drops below $100 at some point.

**Correlated stock returns.** Simulate 1,000 monthly returns for three stocks with means 1%, 1.5% and 0.8%, standard deviations 5%, 8% and 3%, and correlations 0.3 (A-B), 0.1 (A-C) and 0.2 (B-C). Save the result as 'returns' to use it as input data for the portfolio example.

_Answer:_ With 'exact' checked the simulated returns have exactly the means, standard deviations and correlations given.

**Portfolio weights (input data).** Using the simulated 'returns' as input data, which weights for stocks A and B (C gets the rest) give the lowest risk (standard deviation of the portfolio return)? Search weights from 0 to 1 in steps of 0.05.

_Answer:_ The lowest-risk portfolio puts most of the money in the least volatile stock (C) with a small share in A: about 25% A, 0% B and 75% C.

