"""Convert R data files (.rds, .rda) to parquet, keeping their descriptions.

Radiant (R) stores a dataset's markdown description in the ``description``
attribute of the data frame. ``rds_to_parquet`` reads the files with R, carries
every column over with its type (a factor becomes an Enum with R's level
order), and writes a parquet file with the description stored the way
:func:`pyrsm.data.add_description` stores it: inside the file and beside it as
``<name>_description.md``. Radiant-AI and ``pyrsm.load_data`` read both.

Only base R is needed. Columns travel from R as raw little-endian binary, so
numbers arrive exactly as R holds them.

From a terminal::

    rds2parquet ~/data/*.rds
    python -m pyrsm.data.rds ~/data --out-dir ~/parquet
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from pyrsm.data.description import DESCRIPTION_KEY, description_path

#: The R file types converted; a folder is searched for these.
R_SUFFIXES = (".rds", ".rda", ".rdata")

#: Where R installers put Rscript when it is not on the PATH.
_RSCRIPT_LOCATIONS = (
    "/usr/local/bin/Rscript",
    "/usr/bin/Rscript",
    "/opt/homebrew/bin/Rscript",
    "/Library/Frameworks/R.framework/Resources/bin/Rscript",
    "C:/Program Files/R/R-*/bin/Rscript.exe",
    "C:/Program Files/R/R-*/bin/x64/Rscript.exe",
)

R_NOT_FOUND = (
    "R could not be found, and it is needed to read .rds and .rda files. Install R "
    "from https://cran.r-project.org, or tell pyrsm where Rscript is with "
    "rscript=\"/path/to/Rscript\" (or the RSCRIPT environment variable)."
)

# Writes each data frame in the files given as one folder per object: column
# names, type codes and per-column binary values, NA masks, factor levels,
# time zones and the description. Base R only.
_EXPORT_R = r"""
args <- commandArgs(trailingOnly = TRUE)
out <- args[1]
files <- args[-1]

bin_file <- function(path, writer) {
  con <- file(path, "wb")
  on.exit(close(con))
  writer(con)
}
put_str <- function(x, path) {
  x <- enc2utf8(as.character(x))
  x[is.na(x)] <- ""
  bin_file(path, function(con) writeBin(x, con, useBytes = TRUE))
}
put_num <- function(x, path, size) {
  bin_file(path, function(con) writeBin(x, con, size = size, endian = "little"))
}
put_na <- function(miss, path) {
  if (any(miss)) bin_file(path, function(con) writeBin(as.raw(miss), con))
}
put_dbl <- function(v, base, miss = is.na(v) & !is.nan(v)) {
  v <- as.double(v)
  v[miss] <- 0
  put_num(v, base, 8)
  put_na(miss, paste0(base, ".na"))
}
as_text <- function(x) {
  if (is.list(x)) vapply(x, function(e) paste(format(e), collapse = ", "), "")
  else as.character(x)
}

export_col <- function(x, base) {
  if (inherits(x, "POSIXlt")) x <- as.POSIXct(x)
  if (is.factor(x)) {
    lev <- levels(x)
    codes <- as.integer(x)
    if (anyNA(lev)) {
      na_lev <- which(is.na(lev))
      codes[codes %in% na_lev] <- NA
      keep <- seq_along(lev)[-na_lev]
      codes <- match(codes, keep)
      lev <- lev[-na_lev]
    }
    codes[is.na(codes)] <- 0L
    put_num(codes, base, 4)
    put_str(lev, paste0(base, ".levels"))
    return(c("factor", ""))
  }
  if (inherits(x, "Date")) {
    v <- floor(unclass(x))
    put_dbl(v, base, is.na(v))
    return(c("date", ""))
  }
  if (inherits(x, "POSIXct")) {
    tz <- attr(x, "tzone")
    if (is.null(tz) || !nzchar(tz[1])) tz <- Sys.timezone()
    if (length(tz) == 0 || is.na(tz[1])) tz <- "UTC"
    put_str(tz[1], paste0(base, ".tz"))
    put_dbl(unclass(x), base, is.na(x))
    return(c("datetime", ""))
  }
  if (inherits(x, "difftime")) {
    put_dbl(as.double(x, units = "secs"), base)
    return(c("duration", ""))
  }
  if (inherits(x, "integer64")) {
    if (requireNamespace("bit64", quietly = TRUE)) {
      put_dbl(bit64::as.double.integer64(x), base)
      return(c("double", "integer64 values stored as numbers (doubles)"))
    }
    put_str(rep("", length(x)), base)
    return(c("character", "integer64 column left empty: install the bit64 R package"))
  }
  if (is.logical(x) && is.null(dim(x))) {
    v <- as.integer(x)
    miss <- is.na(v)
    v[miss] <- 0L
    put_num(v, base, 4)
    put_na(miss, paste0(base, ".na"))
    return(c("logical", ""))
  }
  if (is.integer(x) && is.null(dim(x))) {
    v <- as.vector(unclass(x))
    miss <- is.na(v)
    v[miss] <- 0L
    put_num(v, base, 4)
    put_na(miss, paste0(base, ".na"))
    return(c("integer", ""))
  }
  if (is.double(x) && is.null(dim(x))) {
    put_dbl(as.vector(unclass(x)), base)
    return(c("double", ""))
  }
  if (is.character(x) && is.null(dim(x))) {
    put_str(x, base)
    put_na(is.na(x), paste0(base, ".na"))
    return(c("character", ""))
  }
  txt <- if (!is.null(dim(x))) apply(as.matrix(x), 1, function(r) paste(r, collapse = ", ")) else as_text(x)
  put_str(txt, base)
  put_na(is.na(txt), paste0(base, ".na"))
  c("character", paste0(class(x)[1], " column stored as text"))
}

export_df <- function(df, dir) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  descr <- attr(df, "description")
  if (is.character(descr) && length(descr) > 0 && any(nzchar(descr))) {
    descr <- paste(enc2utf8(descr), collapse = "\n")
    bin_file(file.path(dir, "description.md"), function(con) writeBin(charToRaw(descr), con))
  }
  writeLines(as.character(nrow(df)), file.path(dir, "nrow"))
  put_str(names(df), file.path(dir, "names"))
  info <- character(0)
  for (j in seq_along(df)) info <- c(info, export_col(df[[j]], file.path(dir, paste0("col", j))))
  put_str(info, file.path(dir, "types"))
}

for (i in seq_along(files)) {
  fdir <- file.path(out, i)
  dir.create(fdir, showWarnings = FALSE)
  status <- tryCatch({
    path <- files[i]
    if (grepl("\\.rds$", path, ignore.case = TRUE)) {
      objs <- list(readRDS(path))
      names(objs) <- ""
    } else {
      env <- new.env()
      objs <- mget(load(path, envir = env), envir = env)
    }
    is_df <- vapply(objs, is.data.frame, logical(1))
    put_str(names(objs)[is_df], file.path(fdir, "objects"))
    for (k in which(is_df)) export_df(objs[[k]], file.path(fdir, sum(is_df[seq_len(k)])))
    if (!any(is_df)) {
      paste0("skip: holds no data frame (", paste(unique(vapply(objs, function(o) class(o)[1], "")), collapse = ", "), ")")
    } else "ok"
  }, error = function(e) paste("error:", conditionMessage(e)))
  writeLines(enc2utf8(status), file.path(fdir, "status"), useBytes = TRUE)
}
"""


def find_rscript(rscript: str | Path | None = None) -> str:
    """The path to ``Rscript``: the one given, ``$RSCRIPT``, the PATH, or a usual install location.

    Raises
    ------
    RuntimeError
        When R cannot be found (the message says how to install it or point to it).
    """
    if rscript is not None:
        found = shutil.which(str(Path(rscript).expanduser()))
        if found is None:
            raise RuntimeError(f"There is no Rscript at {str(rscript)!r}. {R_NOT_FOUND}")
        return found
    for candidate in (os.environ.get("RSCRIPT"), shutil.which("Rscript")):
        if candidate and shutil.which(candidate):
            return shutil.which(candidate)
    for pattern in _RSCRIPT_LOCATIONS:
        for candidate in sorted(glob.glob(pattern), reverse=True):  # newest R version first
            if os.access(candidate, os.X_OK):
                return candidate
    raise RuntimeError(R_NOT_FOUND)


def _r_files(paths: str | Path | list[str | Path]) -> list[Path]:
    """Every R data file in ``paths``: files, folders (their R data files) and glob patterns."""
    if isinstance(paths, str | Path):
        paths = [paths]
    found: list[Path] = []
    for item in paths:
        text = os.path.expanduser(str(item))
        if glob.has_magic(text):
            matches = [Path(m) for m in sorted(glob.glob(text, recursive=True))]
            if not matches:
                raise FileNotFoundError(f"No files match {str(item)!r}.")
            found += [m for m in matches if m.is_file() and m.suffix.lower() in R_SUFFIXES]
        elif Path(text).is_dir():
            found += sorted(p for p in Path(text).iterdir() if p.is_file() and p.suffix.lower() in R_SUFFIXES)
        elif Path(text).is_file():
            if Path(text).suffix.lower() not in R_SUFFIXES:
                raise ValueError(f"{str(item)!r} is not an R data file (.rds, .rda or .RData).")
            found.append(Path(text))
        else:
            raise FileNotFoundError(f"There is no file or folder at {str(item)!r}.")
    unique = list(dict.fromkeys(p.resolve() for p in found))
    if not unique:
        raise FileNotFoundError(f"No .rds, .rda or .RData files found in {[str(p) for p in paths]}.")
    return unique


def _strings(path: Path) -> list[str]:
    """The NUL-terminated UTF-8 strings R's ``writeBin`` wrote."""
    return [s.decode("utf-8", errors="replace") for s in path.read_bytes().split(b"\0")[:-1]]


def _with_nulls(s: pl.Series, base: Path) -> pl.Series:
    mask_file = base.with_name(base.name + ".na")
    if mask_file.is_file():
        missing = np.flatnonzero(np.frombuffer(mask_file.read_bytes(), dtype=np.uint8))
        s = s.scatter(missing, None)
    return s


def _column(name: str, kind: str, base: Path) -> pl.Series:
    """One column, rebuilt with the polars type matching R's."""
    if kind == "factor":
        levels = list(dict.fromkeys(_strings(base.with_name(base.name + ".levels"))))
        codes = pl.Series(np.frombuffer(base.read_bytes(), dtype="<i4").astype(np.int64) - 1)
        idx = codes.set(codes < 0, None)
        return pl.Series(name, levels, dtype=pl.Enum(levels)).gather(idx).alias(name)
    if kind == "character":
        return _with_nulls(pl.Series(name, _strings(base), dtype=pl.String), base)
    if kind in ("integer", "logical"):
        s = _with_nulls(pl.Series(name, np.frombuffer(base.read_bytes(), dtype="<i4").copy()), base)
        return s.cast(pl.Boolean) if kind == "logical" else s
    values = np.frombuffer(base.read_bytes(), dtype="<f8").copy()
    if kind == "double":
        return _with_nulls(pl.Series(name, values), base)
    if kind == "date":
        return _with_nulls(pl.Series(name, values.astype(np.int32)), base).cast(pl.Date)
    micro = _with_nulls(pl.Series(name, np.round(values * 1e6).astype(np.int64)), base)
    if kind == "duration":
        return micro.cast(pl.Duration("us"))
    utc = micro.cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    tz = _strings(base.with_name(base.name + ".tz"))[0]
    try:
        return utc.dt.convert_time_zone(tz)
    except Exception:  # noqa: BLE001 - a time zone polars does not know
        return utc


def _frame(obj_dir: Path) -> tuple[pl.DataFrame, str | None, list[str]]:
    """The data frame R exported to ``obj_dir``, its description and notes."""
    names = _strings(obj_dir / "names")
    info = _strings(obj_dir / "types")
    seen: dict[str, int] = {}
    columns, notes = [], []
    for j, name in enumerate(names):
        kind, note = info[2 * j], info[2 * j + 1]
        if name in seen:  # R allows repeated names; parquet does not
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        seen.setdefault(name, 0)
        columns.append(_column(name, kind, obj_dir / f"col{j + 1}"))
        if note:
            notes.append(f"{name}: {note}")
    frame = pl.DataFrame(columns)
    descr_file = obj_dir / "description.md"
    descr = descr_file.read_text(encoding="utf-8") if descr_file.is_file() else None
    return frame, (descr if descr and descr.strip() else None), notes


def _write(frame: pl.DataFrame, path: Path, descr: str | None) -> None:
    """Write the parquet file (and its ``_description.md``), replacing any earlier one whole."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        frame.write_parquet(str(partial), metadata={DESCRIPTION_KEY: descr} if descr else None)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    if descr:
        description_path(path).write_text(descr, encoding="utf-8")


def rds_to_parquet(
    paths: str | Path | list[str | Path],
    out_dir: str | Path | None = None,
    overwrite: bool = True,
    rscript: str | Path | None = None,
    quiet: bool = False,
) -> list[Path]:
    """Convert R data files to parquet, keeping each dataset's description.

    Parameters
    ----------
    paths : str | Path | list
        R data files (.rds, .rda, .RData), folders holding them, or glob
        patterns such as ``"~/data/*.rds"``; a single one or a list.
    out_dir : str | Path | None
        Folder for the parquet files. By default each is written beside its R file.
    overwrite : bool
        Replace parquet files that already exist (default). With ``False`` they
        are left as they are and reported as skipped.
    rscript : str | Path | None
        Path to ``Rscript`` when R is not on the PATH (also: ``$RSCRIPT``).
    quiet : bool
        Do not print a line per file.

    Returns
    -------
    list[Path]
        The parquet files written.

    Raises
    ------
    RuntimeError
        When R cannot be found.

    Notes
    -----
    An .rds file becomes ``<file name>.parquet``. Each data frame in an .rda
    file becomes ``<object name>.parquet`` (Radiant names the object after the
    file). The markdown in R's ``description`` attribute is stored inside the
    parquet file (metadata key ``description``) and beside it as
    ``<name>_description.md``, as :func:`add_description` does.

    Types: factor -> Enum (R's level order), character -> String,
    integer -> Int32, double -> Float64 (``NA`` -> null, ``NaN`` kept),
    logical -> Boolean, Date -> Date, POSIXct -> Datetime in the column's
    time zone, difftime -> Duration. Anything else (list columns, matrices) is
    stored as text and reported.

    Examples
    --------
    >>> rsm.data.rds_to_parquet("~/data/*.rds")
    >>> rsm.data.rds_to_parquet(["data/diamonds.rda", "data/titanic.rds"], out_dir="parquet")
    >>> rsm.data.read_description("parquet/diamonds.parquet")
    """
    files = _r_files(paths)
    exe = find_rscript(rscript)
    target = None if out_dir is None else Path(out_dir).expanduser()
    say = (lambda msg: None) if quiet else print

    written: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="pyrsm-rds-") as tmp:
        script = Path(tmp) / "export.R"
        script.write_text(_EXPORT_R, encoding="utf-8")
        export = Path(tmp) / "out"
        export.mkdir()
        result = subprocess.run(
            [exe, "--vanilla", str(script), str(export), *[str(f) for f in files]],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(f"R failed while reading the files:\n{(result.stderr or result.stdout).strip()}")

        for i, src in enumerate(files, start=1):
            fdir = export / str(i)
            status_file = fdir / "status"
            status = status_file.read_text(encoding="utf-8").strip() if status_file.is_file() else "error: R did not read this file"
            if status != "ok":
                say(f"{src.name}: {status.replace('error:', 'not converted:', 1).replace('skip:', 'skipped,', 1)}")
                continue
            objects = _strings(fdir / "objects")
            for k, obj in enumerate(objects, start=1):
                name = obj or src.stem
                dest = (target or src.parent) / f"{name}.parquet"
                if dest.exists() and not overwrite:
                    say(f"{src.name}: skipped, {dest} exists (overwrite=False)")
                    continue
                frame, descr, notes = _frame(fdir / str(k))
                _write(frame, dest, descr)
                written.append(dest)
                rows, cols = frame.shape
                extra = "with description" if descr else "no description"
                say(f"{src.name} -> {dest} ({rows:,} rows, {cols} columns, {extra})")
                for note in notes:
                    say(f"  note: {note}")
    return written


def main(argv: list[str] | None = None) -> int:
    """``rds2parquet``: the command-line front end of :func:`rds_to_parquet`."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="rds2parquet",
        description="Convert R data files (.rds, .rda) to parquet, keeping the markdown dataset description.",
    )
    parser.add_argument("paths", nargs="+", help='R data files, folders, or patterns such as "~/data/*.rds"')
    parser.add_argument("-o", "--out-dir", help="folder for the parquet files (default: beside each R file)")
    parser.add_argument("--no-overwrite", action="store_true", help="leave existing parquet files as they are")
    parser.add_argument("--rscript", help="path to Rscript when R is not on the PATH")
    args = parser.parse_args(argv)
    try:
        written = rds_to_parquet(args.paths, out_dir=args.out_dir, overwrite=not args.no_overwrite, rscript=args.rscript)
    except (RuntimeError, FileNotFoundError, ValueError) as err:
        print(f"rds2parquet: {err}", file=sys.stderr)
        return 1
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
