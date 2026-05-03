# Transcript — diamonds linear vs log-log regression

## User request

> "can you regress price on carat and clarity for me? data is at
> /Users/vnijs/gh/pyrsm/examples/data/data/diamonds.parquet . my TA said the
> residuals will look bad and i should try a log-log version, can you check?"

The user provided an absolute path to the parquet, a clear `rvar` (price), a
clear `evar` list (carat, clarity), and an explicit invitation to walk the
residual diagnostics + log-log refit branch of the workflow. So Step 1
("ask for the data location") is already satisfied.

## Flow

1. **Step 1 — Data location.** Already given as an absolute path.
2. **Step 2 — Load.** Parquet has 3,000 rows / 11 columns. Sidecar
   description file `diamonds_description.md` was found and read. Useful
   facts from it:
   - price in US dollars (\$338--\$18,791)
   - carat in weight (0.20--3.00)
   - clarity ordered worst-to-best: I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF
   - I1 is the worst, IF (Internally Flawless) is the best
3. **Step 3 — Specification.** Locked in by the user:
   - rvar = `price`
   - evar = `["carat", "clarity"]`
   - clarity is an Enum already; reference level will be I1 (alphabetically
     first / worst clarity grade), which is the natural baseline anyway.
4. **Step 4 — Fit linear model** `price ~ carat + clarity`.
   - All 8 coefficients significant at p < 0.001.
   - R^2 = 0.904, F(8, 2991) = 3530, RMSE = $1,224.
   - Per-carat dollar slope: \$8,438. Clarity premia in dollars range from
     ~\$2,791 (SI2 vs I1) to ~\$5,265 (IF vs I1).
5. **Step 5 — Interpretation of the linear fit.** Done in
   `explanation.md`. Crucially, no variable is dropped on a p-value basis
   (all are p < 0.001 anyway, so the OVB protocol from Step 6 doesn't even
   come up).
6. **Step 7 — Residual dashboard for the linear spec.** Saved as
   `dashboard_linear.png`. Three textbook problems:
   - Predicted-vs-actual is curved (missed non-linearity).
   - Residuals-vs-predicted shows a megaphone (heteroscedasticity).
   - Q-Q plot has heavy right tail (non-normal residuals).
   The TA was right.
7. **Refit log-log** `ln(price) ~ ln(carat) + clarity`.
   - All 8 coefficients still significant at p < 0.001.
   - R^2 = 0.966, RMSE = 0.187 in log-units.
   - carat_ln coefficient = 1.809 — that's the carat **elasticity**.
   - Clarity dummies translated to exact %-changes via 100*(exp(β)-1)%:
     +56%, +81%, +111%, +121%, +157%, +175%, +195% for SI2, SI1, VS2, VS1,
     VVS2, VVS1, IF respectively (vs I1 baseline).
8. **Residual dashboard for the log-log spec.** Saved as
   `dashboard_loglog.png`. The curvature is gone, the funnel is gone, the
   Q-Q plot is much closer to a straight line. Real improvement, not
   cosmetic.

## Files written to outputs/

- `analysis.py` — full reproducible script.
- `analysis_output.txt` — captured stdout from running it.
- `dashboard_linear.png` — six-panel diagnostic for the linear spec.
- `dashboard_loglog.png` — six-panel diagnostic for the log-log spec.
- `explanation.md` — pedagogical writeup of the two specs, the residual
  problems, the elasticity, and the exact %-change table for the clarity
  dummies.
- `transcript.md` — this file.

## Notes / quirks

- The linear `df.select(...)` summary calls initially errored because polars
  refuses two `pl.col("price")...` expressions in one `select` (duplicate
  output names). Added `.alias()` calls and re-ran cleanly.
- Both dashboards saved on the first try via plotnine's `g.save(...)` with
  the Agg matplotlib backend.
- R^2 (0.904 vs 0.966) is *not* a fair head-to-head metric since the
  response variable changed (price vs ln(price)). The dashboards are the
  right comparison. The writeup says so explicitly so the student doesn't
  walk away thinking R^2 alone justifies the log-log model.
