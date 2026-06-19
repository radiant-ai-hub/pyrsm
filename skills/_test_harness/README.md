# pyrsm-skill transcript test harness

A small backend-agnostic check that flags the three pyrsm-skill failure modes
in a model transcript (works against Pi, Claude, GPT, or any other model output).

## Failure modes checked

1. **NO_BAD_SUMMARY_FLAGS** — The transcript should not be calling
   `reg.summary(rmse=True, ssq=True)` or similar UNLESS the user prompt
   explicitly asked for sum-of-squares, RMSE, VIF, or coefficient CIs. The
   default is plain `summary()`.

2. **FULL_OUTPUT_SHOWN** — When `.summary()` is called, the transcript must
   include the verbatim printed output (the polars-style table with `┌`,
   `│`, `└` chars AND keywords like `p.value`, `Adjusted R-squared`,
   `Nr obs`, `Chi-squared`). Abbreviation phrases like "omitted for brevity",
   "high-level summary", or "main takeaways" trigger a fail.

3. **NOT_LAME_BULLETS** — The interpretation should be detailed (real-world
   units, "holding constant" qualifier, H0/Hₐ statements, significance
   verdicts at the 5% level), not a generic 4-bullet "main takeaways" list
   like:

   > Main takeaways:
   > - carat is strongly positively associated with price.
   > - Better clarity levels are associated with higher prices.
   > - depth and table are not statistically significant.

   The check fires when it detects this exact pattern (a `Main takeaways:`
   header + 3+ terse bullets + few detailed-interpretation markers).

4. **IMPORTANCE_DISCLAIMER_PRESENT** — Whenever the transcript makes a
   relative-importance claim (e.g. "X was most strongly associated", "the
   bottom line is that A matters most", "Y is the biggest predictor"),
   the response MUST close with a disclaimer that:
   - References permutation importance (PIP) by name or by `plot("pip")`, AND
   - Notes the scale-comparability caveat (coefficients / odds ratios / PDPs
     are NOT directly comparable across predictors on different scales).

   This catches the Pi-style failure: *"The PDPs suggest survival was most
   strongly associated with sex..."* — interpreting PDP steepness as
   relative importance, which is wrong (steeper PDPs can just mean larger
   natural ranges, not larger contributions to model fit). The 5 model
   skills (regress, logistic, rforest, xgboost, mlp) now require this
   disclaimer at the end of any interpretation, even when the user did
   not explicitly ask "which variable matters most".

## Usage

```bash
# Check a transcript file
python check_transcript.py /path/to/transcript.txt --skill pyrsm-regress

# Pipe a transcript in from another tool (e.g. a pi-cli wrapper)
pi-cli run-skill pyrsm-regress --prompt "fit regress on diamonds…" | \
    python check_transcript.py - --skill pyrsm-regress

# Provide the user prompt to allow summary(rmse=True) when the user asked
python check_transcript.py transcript.txt --prompt prompt.txt --skill pyrsm-regress

# JSON output for CI integration
python check_transcript.py transcript.txt --json
```

Exit code is 0 on pass, 1 on fail (any check failed), 2 on usage error.

## Self-test

Run the bundled self-test to confirm the harness works:

```bash
bash run_self_test.sh
```

The self-test exercises three fixtures:

- `fixtures/good_regress_transcript.txt` — should PASS all three checks.
- `fixtures/bad_lame_bullets_transcript.txt` — should FAIL (summary flags
  + truncated output + "main takeaways" bullets).
- `fixtures/bad_truncated_transcript.txt` — should FAIL ("omitted for
  brevity" + bad summary flags).

If the self-test passes, the checker is working correctly. If you change
the patterns in `check_transcript.py`, re-run the self-test to make sure
you haven't broken the existing classifications.

## Adding new checks or skills

`check_transcript.py` is intentionally a single file with regex-based checks
so you can edit it directly. To add a new failure-mode check:

1. Add the regex pattern(s) at the top of the file alongside the existing
   ones (`BAD_SUMMARY_FLAG_RE`, `TABLE_CHARS_RE`, etc.).
2. Add a new function `check_<name>(transcript, user_prompt) -> CheckResult`.
3. Append it to the list returned by `run_checks()`.
4. Add a fixture file under `fixtures/` that demonstrates the failure mode.
5. Extend `run_self_test.sh` to test the new fixture.

## Integration with the existing eval workspace

The eval workspace under `skills/pyrsm-<skill>-workspace/iteration-N/` already
stores transcripts as `<test-name>/with_skill/outputs/transcript.md`. You can
batch-check all of them:

```bash
for transcript in skills/pyrsm-regress-workspace/iteration-*/*/with_skill/outputs/transcript.md; do
    echo "=== $transcript ==="
    python skills/_test_harness/check_transcript.py "$transcript" --skill pyrsm-regress
    echo
done
```

This makes it easy to retest after a skill change without re-running the
upstream eval harness.
