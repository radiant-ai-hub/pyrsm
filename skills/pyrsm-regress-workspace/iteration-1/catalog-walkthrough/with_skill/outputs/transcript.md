# Conversational flow

**User opens with:** "I have a dataset of catalog customer sales at /Users/vnijs/gh/pyrsm/examples/data/model/catalog.parquet. Predict Sales from Income, HH_size, Age. Walk me through it like my prof would. I'm new to regression."

### Step 1 — Confirm the data location
- Path is already absolute and the file exists, so no need to ping the user for it. Skipped straight to loading.

### Step 2 — Probe the data + read the sidecar description
- Loaded `catalog.parquet` with polars and confirmed shape (200, 5) and dtypes.
- Found the sidecar `catalog_description.md` next to the data file and read it. Key takeaways for interpretation:
  - Sales is in dollars.
  - Income is measured in **thousands of dollars** (this matters for the coefficient interpretation).
  - HH_size is number of people; Age is years.
  - 200 households, random sample from the company database.

### Step 3 — Propose the specification
- The user already gave the spec in their question, so I confirmed it back to them: response = `Sales`, predictors = `Income`, `HH_size`, `Age`. Linear regression. No interactions. No log transforms (yet).
- Quick sanity check: Sales is numeric (good for OLS), no missing values to worry about, household sales is the kind of right-skew variable where I'd flag a log option later if residuals look bad — but defaulting to linear since that's what the user asked for and it's the standard textbook starting point.

### Step 4 — Fit the model
- Wrote `analysis.py`: load with polars, fit with `rsm.model.regress({"catalog": df}, rvar="Sales", evar=["Income", "HH_size", "Age"])`, summary with `rmse=True, ssq=True`.
- Ran it and captured stdout to `analysis_output.txt`.

### Step 5 — Interpret the output (this is where most of the value is)
Walked the user through, in this order:
1. **F-test** on the model as a whole. Stated H0 / Ha explicitly. F = 32.33 on (3, 196), p < 0.001 → reject H0; the model is doing real work.
2. **R² = 0.331** → 33.1% of variance in Sales explained. Mentioned RMSE = \$88 to give a feel for prediction noise.
3. **Each coefficient**, with units anchored to the description file:
   - Income: +\$1.78 in Sales per \$1,000 of income — significant (p < 0.001).
   - HH_size: +\$22.12 in Sales per additional household member — significant (p < 0.001).
   - Age: +\$0.45 per year — **not** significant (p = 0.559).
   - Intercept noted as not meaningfully interpretable.
4. Hammered on the **"holding all other variables constant"** qualifier — distinguished it from a simple scatter plot.
5. Wrote a plain-English bottom-line: Income and HH_size matter, Age doesn't, model overall is significant but explains a third of the variance.

### Step 6 — Offer (don't run) diagnostics
- Per skill guidance, I did not auto-run residual plots. I closed by listing the four diagnostics I would run if asked (`dist`, `corr`, `dashboard`, `residual`) and what each one is for, and offered to generate scenario predictions with `reg.predict(cmd=...)`.
- Left it to the user to decide whether to continue into diagnostics or stop here.
