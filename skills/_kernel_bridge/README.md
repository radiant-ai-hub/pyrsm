# pyrsm-kernel bridge

A long-lived IPython kernel for pyrsm analysis workflows that lets your coding
agent (Pi, Claude Code, or anything that can run shell commands) keep state
across turns: load data once, fit a model once, then run many follow-up cells
against the cached objects.

## Why this exists

Most coding-agent CLIs run Python via subprocess. Every `python -c '...'` or
`python script.py` call is a **fresh interpreter**: loaded DataFrames and
fitted models do not survive between agent turns. For pyrsm work this is
especially painful — re-loading a 1 GB parquet and re-fitting a Random Forest
on every turn wastes minutes.

With this bridge, the kernel runs as a separate process. The agent doesn't
own it. Each agent turn sends a code snippet via `pyrsm-kernel exec`; the
kernel evaluates and returns output. The kernel keeps all in-memory state
(modules, DataFrames, fitted models, plotnine plot objects) between turns.

## Installation

The wrapper is a single Python file at
`<repo>/skills/_kernel_bridge/pyrsm-kernel`. It depends only on `jupyter_client`
and `ipykernel` (both stdlib of the Jupyter ecosystem). Confirm they're
available in your Python environment:

```bash
python3 -c "import jupyter_client, ipykernel; print('ok')"
```

If you want to call `pyrsm-kernel` from anywhere, symlink it onto your `PATH`:

```bash
ln -s "$(pwd)/skills/_kernel_bridge/pyrsm-kernel" ~/.local/bin/pyrsm-kernel
# or pick another bin dir on your PATH
```

## Daily workflow

```bash
# 1. Activate the venv with pyrsm installed and start a kernel for this project.
cd /path/to/project
source .venv/bin/activate
pyrsm-kernel start                  # name defaults to the cwd's basename

# 2. Sanity check:
pyrsm-kernel status
pyrsm-kernel exec 'import pyrsm; print(pyrsm.__version__)'

# 3. Now ask your agent to use the kernel for all Python work.
#    With Pi: add the SYSTEM_kernel.md fragment to your .pi config so Pi
#    routes Python calls through `pyrsm-kernel exec` instead of `python -c`.

# ... do your analysis ...

# 4. When done (or to start fresh), stop the kernel:
pyrsm-kernel stop
```

## Commands

```
pyrsm-kernel start [--name NAME] [--force]
    Start a kernel. NAME defaults to the basename of the current directory.
    If a kernel with the same name is already running, refuses unless --force
    (which stops the old one and starts a new one).

pyrsm-kernel exec [--name NAME] '<CODE>'
pyrsm-kernel exec [--name NAME] -                   # read code from stdin
    Send CODE to the running kernel. Streams stdout, stderr, and any
    text/plain display output back to this process's stdout. Exits non-zero
    if the kernel raises an exception (the traceback is printed).
    Options:
      --startup-timeout SECONDS    (default 10)
      --cell-timeout    SECONDS    (default 600)

pyrsm-kernel status [--name NAME]
    Report whether the named kernel is alive, its pid, and start time.

pyrsm-kernel stop [--name NAME]
    Kill the named kernel and clean up its cache files.

pyrsm-kernel list
    List all running pyrsm-kernels.
```

## Cache layout

State for each named kernel lives under `$XDG_CACHE_HOME/pyrsm-kernel/<name>/`
(typically `~/.cache/pyrsm-kernel/<name>/`):

- `connection.json` — Jupyter connection info (ports, key, signature scheme).
- `kernel.pid`      — PID of the kernel process.
- `kernel.log`      — combined stdout/stderr from the kernel itself.
- `started.txt`     — ISO timestamp when the kernel was launched.

The kernel inherits the environment in which `pyrsm-kernel start` ran. Always
activate your project venv before starting the kernel.

## Working with plots

Plotnine and matplotlib figures returned by `.plot()` methods are **objects in
memory**. The kernel can hold them between turns, but to make a plot visible
to your agent (and to you in the terminal), save it to disk first:

```bash
pyrsm-kernel exec '
p = reg.plot("pip")
p.save("pip.png", width=8, height=5, dpi=120)
print("saved to pip.png")
'
```

Then the agent can read the PNG with its `read` tool (Pi, Claude Code) or
display it inline.

## Working with cached models across plots / predictions / evaluations

This is the whole point — the kernel keeps everything alive:

```bash
# Turn 1: load + fit
pyrsm-kernel exec '
import polars as pl, pyrsm as rsm
df = pl.read_parquet("/abs/path/diamonds.parquet")
reg = rsm.model.regress({"diamonds": df}, rvar="price", evar=["carat", "clarity"])
reg.summary()
'

# Turn 2: predict with the cached `reg` (no re-fit!)
pyrsm-kernel exec 'pred = reg.predict(); pred.head()'

# Turn 3: plot residual dashboard (still using cached `reg`)
pyrsm-kernel exec '
p = reg.plot("dashboard")
p.save("dashboard.png", width=10, height=8, dpi=120)
print("saved")
'

# Turn 4: do a counterfactual scenario
pyrsm-kernel exec 'reg.predict(cmd={"carat": [0.5, 1.0, 1.5, 2.0]})'
```

Each `exec` is a separate agent-tool call but reuses the same kernel, so
`reg`, `df`, and everything else stays in memory.

## Pi integration

For Pi, drop the bundled `SYSTEM_kernel.md` into your project's `.pi/`
directory (or append it to `~/.pi/SYSTEM.md` for all projects). It tells Pi
to route Python code through `pyrsm-kernel exec`:

```bash
# Per-project
mkdir -p .pi
cp /path/to/skills/_kernel_bridge/SYSTEM_kernel.md .pi/SYSTEM.md
# Or: append to a global Pi system prompt
cat /path/to/skills/_kernel_bridge/SYSTEM_kernel.md >> ~/.pi/SYSTEM.md
```

Then start a kernel and start `pi` in the same project. Pi will automatically
use `pyrsm-kernel exec` for Python work.

## Working with Claude Code

For Claude Code, drop the same `SYSTEM_kernel.md` as `CLAUDE.md` in your
project root, or append it to your existing CLAUDE.md. Claude will see the
instructions and use the kernel for Python work.

```bash
cat /path/to/skills/_kernel_bridge/SYSTEM_kernel.md >> CLAUDE.md
```

## Troubleshooting

**"kernel '...' not running"** — Run `pyrsm-kernel start` first.

**"kernel '...' is stale (pid not running)"** — Kernel crashed or got killed.
Run `pyrsm-kernel start --force` to clean up and start fresh.

**"timed out after Xs"** — A long-running cell. Bump `--cell-timeout`, or
interrupt the kernel with Ctrl-C (which sends SIGINT — most pyrsm calls
respect it; CV grid searches may not).

**A loaded dataset is huge and slowing the kernel** — `del df; import gc;
gc.collect()` inside an exec call frees it.

**Want to start clean** — `pyrsm-kernel stop` then `pyrsm-kernel start`.
The log file is preserved at `~/.cache/pyrsm-kernel/<name>/kernel.log` for
postmortem; `rm -rf ~/.cache/pyrsm-kernel/<name>` to nuke it entirely.
