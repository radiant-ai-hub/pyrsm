# Dropping Age from the catalog Sales regression

Short answer: in this particular case, yes, dropping Age is fine — but I want to flag the reasoning, because "p > 0.05, therefore drop it" is a habit that will get you into trouble on other datasets.

## What the two models look like

Full model (`Sales ~ Income + HH_size + Age`):

| term      | coef    | p      |
| --------- | ------- | ------ |
| Intercept | 45.36   | 0.294  |
| Income    | 1.7754  | <0.001 |
| HH_size   | 22.122  | <0.001 |
| Age       | 0.4492  | 0.559  |

Reduced model (`Sales ~ Income + HH_size`):

| term      | coef    | p      |
| --------- | ------- | ------ |
| Intercept | 67.22   | 0.002  |
| Income    | 1.7838  | <0.001 |
| HH_size   | 21.958  | <0.001 |

- R^2: 0.331 -> 0.330 (essentially unchanged)
- Adjusted R^2: 0.321 -> 0.323 (tiny improvement)
- Income coefficient: 1.7754 -> 1.7838 (moves by ~0.5%)
- HH_size coefficient: 22.12 -> 21.96 (moves by <1%)

So, mechanically, the reduced model is the cleaner one to report.

## Why I want to push back on the reasoning, though

A high p-value alone is not a good reason to drop a variable. The thing you really care about when removing a control variable is: **does removing it change the coefficients on the variables I do care about?** That is the omitted-variable-bias question, and it is independent of the dropped variable's own p-value.

Here is why dropping Age is safe in *this* dataset:

- Age is barely correlated with Income (r = 0.065) and HH_size (r = -0.066).
- Age is barely correlated with Sales itself (r = 0.045).

When the dropped variable is uncorrelated with both the outcome and the remaining regressors, you get essentially no bias on the surviving coefficients — and that is exactly what we see (Income and HH_size move by under 1%).

If, instead, Age had been strongly correlated with Income or HH_size, dropping it could have meaningfully shifted those coefficients even though its own p-value was 0.559. In that scenario the "right" answer is usually to keep it as a control, even though it looks "useless," because it is doing real work for the interpretation of the others.

## What I'd actually recommend reporting

1. Report the reduced model (`Sales ~ Income + HH_size`) as your headline model — it is more parsimonious and adjusted R^2 ticks up.
2. In a footnote or appendix, note that you also estimated the model including Age, that Age was not statistically significant (p = 0.56), and that Income and HH_size coefficients were essentially identical in both specifications. That is the honest, reviewer-proof way to present it.

The reduced fit is in `analysis_output.txt`; the script is `analysis.py`.
