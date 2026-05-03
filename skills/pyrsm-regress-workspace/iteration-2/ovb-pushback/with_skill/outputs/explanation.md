# Should you drop Age because p = 0.559?

Short answer: **I would push back on this, and after running the check I'd recommend keeping Age in the model you report.** Let me explain why, then walk you through the check that supports the recommendation.

## Why "high p-value" is not a sufficient reason to drop

A non-significant coefficient is one of the most common triggers for an unforced error in applied regression. Three things to keep in mind:

1. **Non-significance is not the same as "no effect."** A p-value of 0.559 says we don't have enough evidence in *this* sample of 200 customers to distinguish Age's coefficient from zero. It does not say Age is irrelevant. Dropping it discards information.
2. **Dropping a variable can introduce omitted variable bias (OVB) on the coefficients you keep.** If Age is correlated with Income or HH_size and has any real partial effect on Sales, the Income and HH_size coefficients in the reduced model will absorb part of that effect. The numbers and the interpretation both change — and not for the better.
3. **The right time to think about simplification is *after* the model is essentially finalized**, not after the first fit. Even then, simplification should be motivated by something concrete (deployment cost, dimensionality, communication), not just a high p-value.

With **n = 200 and only 3 predictors**, there is no dimensionality problem and no deployment cost worth optimizing. So the honest answer to "should I simplify?" here is: there's no real reason to.

## But let's run the OVB check anyway — that's the protocol

Even when I think simplification isn't warranted, I want you to *see* what the check does, because the same machinery saves you from much worse mistakes on bigger models. The protocol is: snapshot the full-model coefficients, drop the highest-p-value variable, refit, and compare the slope coefficients on the kept variables. Specifically I'm watching for any of three triggers on Income and HH_size:

- `|percent shift| > 10%`, OR
- a **sign flip** (positive becomes negative or vice versa), OR
- a **significance status flip** across the 5% line.

If none of those fire, the dropped variable was approximately orthogonal to the rest of the model and the drop is OVB-safe (in the statistical sense). If any of them fire, I owe you a remediation menu.

### Baseline (full model: Sales ~ Income + HH_size + Age)

| Variable  | Coefficient | p-value     |
|-----------|-------------|-------------|
| Intercept | 45.359      | 0.294       |
| Income    | 1.775       | < .001 ***  |
| HH_size   | 22.122      | < .001 ***  |
| Age       | 0.449       | 0.559       |

R-squared = 0.331, Adjusted R-squared = 0.321, F(3, 196) = 32.3, p < .001, n = 200.

### After dropping Age (Sales ~ Income + HH_size)

| Variable  | Coefficient | p-value     |
|-----------|-------------|-------------|
| Intercept | 67.218      | 0.002 **    |
| Income    | 1.784       | < .001 ***  |
| HH_size   | 21.958      | < .001 ***  |

R-squared = 0.330, Adjusted R-squared = 0.323.

### Percent shifts on the kept slope coefficients

| Variable  | Old coef | New coef | % shift | Sign flip? | Significance flip? |
|-----------|----------|----------|---------|------------|--------------------|
| Income    | 1.7754   | 1.7838   | +0.47%  | no         | no                 |
| HH_size   | 22.122   | 21.958   | −0.74%  | no         | no                 |

**No OVB triggers fire.** Both shifts are well under 1% in magnitude, both signs hold, both keep their significance.

The reason is visible in the predictor correlation matrix:

```
         Income  HH_size    Age
Income    1.000    0.014    0.065
HH_size   0.014    1.000   -0.066
Age       0.065   -0.066    1.000
```

Age's correlations with Income (0.065) and HH_size (−0.066) are both essentially zero. That's what "approximately orthogonal" means in practice — and it's *why* dropping Age barely moves the other two coefficients. OVB requires correlation between the dropped predictor and the kept predictors, and that correlation isn't really there in this sample.

## So what should you actually do?

I'd recommend you **report the full model** — Sales ~ Income + HH_size + Age — and in your write-up state plainly:

> Age has an estimated coefficient of 0.449 with p = 0.559. We fail to reject the null that Age's coefficient is zero at the 5% level. We retain Age in the model because (a) non-significance does not establish absence of effect, and (b) we ran an OVB check by refitting without Age and the Income and HH_size coefficients shifted by less than 1%, so the choice does not materially affect the other coefficients' interpretations.

That paragraph is *more* impressive on a class assignment than a "cleaned-up" two-predictor model would be, because it shows you understand what the p-value means and what it doesn't.

## What if the check *had* fired a trigger?

I'm including this so the protocol is complete in your head for the next dataset. If any kept coefficient had shifted by more than 10%, flipped sign, or flipped significance, I would have stopped and presented you four live options:

1. **Re-include the dropped variable.** The safest default. The cost of carrying a non-significant predictor is small; the cost of a biased kept coefficient is usually much larger.
2. **Relabel a remaining variable.** If Income's coefficient had clearly absorbed part of Age's effect, I'd reframe the interpretation: "Income here is now picking up an income-and-life-stage effect," and write that explicitly into the report.
3. **Combine the correlated variables** into a single composite predictor (a weighted average, an index, a principal component, or a domain-specific construct) and use that in place of either original.
4. **Keep the reduced model, but acknowledge the bias** — only legitimate when you can articulate *why* the bias is acceptable for the question at hand, and you state it explicitly rather than glossing over it.

In our case none of that triggered, so we don't have to choose. But it's the same toolkit you'll want when you fit a bigger model and the shifts come out to 30% or 80%.

## Bottom line

- The p = 0.559 on Age is **not** a license to drop it.
- The OVB check confirms Age is approximately orthogonal to Income and HH_size in this sample, so the drop would be statistically safe — but with n = 200 and 3 predictors, there's no real reason to simplify in the first place.
- Report the full three-predictor model. Mention the OVB check as evidence that the high p-value on Age does not contaminate the Income and HH_size interpretations. That is a stronger writeup than a "clean" reduced model.
