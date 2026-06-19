# Persistent Python kernel — use it for ALL Python work

A long-lived IPython kernel is available in this project via the
`pyrsm-kernel` command. It persists state across your turns: loaded
DataFrames, fitted models, and any other in-memory objects stay alive
between bash calls. **Use it for every Python execution.**

## Rules

1. **Do NOT run Python via `python -c '...'`, `python script.py`, or
   `uv run python ...`** for analysis work. Those start fresh interpreters
   and lose all in-memory state (loaded data, fitted models). They are the
   exact source of the "re-loading data and re-fitting models on every
   turn" problem this kernel is here to solve.

2. **Send code via `pyrsm-kernel exec '<CODE>'` instead.** Example:

   ```bash
   pyrsm-kernel exec '
   import polars as pl
   import pyrsm as rsm
   df = pl.read_parquet("/abs/path/to/data.parquet")
   reg = rsm.model.regress({"name": df}, rvar="y", evar=["x1", "x2"])
   reg.summary()
   '
   ```

3. **State persists across `exec` calls** in the same kernel. After the
   above, you can do:

   ```bash
   pyrsm-kernel exec 'pred = reg.predict(); pred.head()'
   ```

   without re-loading `df` or re-fitting `reg`.

4. **Quote / escape carefully.** `exec` takes a single argument. Use a
   single-quoted heredoc-style block as above. If your code contains
   single quotes, pipe in from stdin instead:

   ```bash
   pyrsm-kernel exec - <<'PYEOF'
   df.filter(pl.col("x") > 0).head()
   PYEOF
   ```

5. **Before the first `exec`, check the kernel is running:**

   ```bash
   pyrsm-kernel status
   ```

   If it returns "not running", start it with `pyrsm-kernel start`. Tell
   the user if you have to start the kernel for them — they may have an
   active session you'd be overwriting.

6. **Plots and figures.** Plotnine and matplotlib figures returned by
   `.plot()` are objects in memory. To make them visible, save to disk:

   ```bash
   pyrsm-kernel exec '
   p = reg.plot("pip")
   p.save("pip.png", width=8, height=5, dpi=120)
   print("saved pip.png")
   '
   ```

   Then read the PNG with your `read` tool to display.

7. **Show the FULL `exec` output verbatim in your response.** The output
   from `pyrsm-kernel exec` is the verbatim stdout from the kernel —
   coefficient tables, summary blocks, predictions. Do NOT abbreviate,
   summarize, or replace it with "main takeaways" bullet lists. See the
   skill-specific Output policy for the details and rationale.

8. **Use plain `.summary()` by default.** Do NOT pass `rmse=True`, `ssq=True`,
   `vif=True`, or `ci=True` unless the user explicitly asks for those.

9. **Errors.** `pyrsm-kernel exec` exits non-zero on uncaught exceptions
   in the kernel and writes the traceback to stderr. Surface the
   traceback in your reply — do not swallow it.

## When NOT to use the kernel

- One-off file probing (`ls`, `cat`, `head`) — just use bash directly.
- The kernel-bridge wrapper itself (e.g., installing dependencies) —
  those go in a fresh subprocess.
- Anything that needs a fresh environment to test (e.g., verifying that
  a fresh install of a package works).

## Quick reference

```
pyrsm-kernel start                  # start kernel for this project (name = cwd basename)
pyrsm-kernel status                 # is the kernel alive?
pyrsm-kernel exec '<code>'          # run code, persisting state
pyrsm-kernel exec - <<'PYEOF'       # multi-line / quote-heavy code via stdin
<code>
PYEOF
pyrsm-kernel stop                   # kill the kernel cleanly
pyrsm-kernel list                   # see all running kernels
```
