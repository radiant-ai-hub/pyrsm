"""rds_to_parquet: R data files to parquet, types and markdown description kept."""

import shutil
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

import pyrsm as rsm
from pyrsm.data import rds as rds_mod

RSCRIPT = shutil.which("Rscript")
needs_r = pytest.mark.skipif(RSCRIPT is None, reason="R (Rscript) is not installed")

DESCR = "## Test data\n\nA *small* dataset.\n\n- x = a number\n"


def _r(code: str) -> None:
    subprocess.run([RSCRIPT, "--vanilla", "-e", code], check=True, capture_output=True)


@pytest.fixture
def typed_rds(tmp_path: Path) -> Path:
    path = tmp_path / "typed.rds"
    _r(f"""
    df <- data.frame(
      dbl = c(1.5, NA, NaN, Inf, 1/3),
      int = c(1L, NA, 3L, -4L, 2147483647L),
      lgl = c(TRUE, NA, FALSE, TRUE, FALSE),
      chr = c("a", NA, "caf\\u00e9", "two\\nlines", ""),
      fct = factor(c("low", "high", NA, "mid", "low"), levels = c("low", "mid", "high", "unused")),
      ord = factor(c("b", "a", "b", "c", "a"), levels = c("c", "b", "a"), ordered = TRUE),
      day = as.Date(c("2024-01-31", NA, "1969-12-31", "2000-02-29", "2024-12-25")),
      when = as.POSIXct(c("2024-03-10 12:30:00", NA, "1970-01-01 00:00:00", "2024-07-04 08:00:00", "2024-12-31 23:59:59"), tz = "America/Los_Angeles"),
      wait = as.difftime(c(1, 2.5, NA, 0, 60), units = "mins"),
      stringsAsFactors = FALSE
    )
    df$lst <- list(1:2, "a", NULL, TRUE, c(1.5, 2))
    attr(df, "description") <- "{DESCR.encode("unicode_escape").decode()}"
    saveRDS(df, "{path}")
    """)
    return path


@needs_r
def test_types_values_and_description_carry_over(typed_rds, capsys):
    (out,) = rsm.data.rds_to_parquet(typed_rds)
    assert out == typed_rds.with_suffix(".parquet")
    df = pl.read_parquet(out)
    assert df.schema["dbl"] == pl.Float64
    assert df.schema["int"] == pl.Int32
    assert df.schema["lgl"] == pl.Boolean
    assert df.schema["chr"] == pl.String
    assert df.schema["fct"] == pl.Enum(["low", "mid", "high", "unused"])  # R's level order, unused kept
    assert df.schema["ord"] == pl.Enum(["c", "b", "a"])
    assert df.schema["day"] == pl.Date
    assert df.schema["when"] == pl.Datetime("us", "America/Los_Angeles")
    assert df.schema["wait"] == pl.Duration("us")
    assert df.schema["lst"] == pl.String

    dbl = df["dbl"].to_list()
    assert dbl[0] == 1.5 and dbl[1] is None and dbl[2] != dbl[2] and dbl[3] == float("inf")
    assert dbl[4] == 1 / 3  # exact: binary transfer, no rounding through text
    assert df["int"].to_list() == [1, None, 3, -4, 2147483647]
    assert df["lgl"].to_list() == [True, None, False, True, False]
    assert df["chr"].to_list() == ["a", None, "café", "two\nlines", ""]
    assert df["fct"].to_list() == ["low", "high", None, "mid", "low"]
    assert df["day"].to_list() == [date(2024, 1, 31), None, date(1969, 12, 31), date(2000, 2, 29), date(2024, 12, 25)]
    when = df["when"].to_list()
    assert when[1] is None
    assert when[0].replace(tzinfo=None) == datetime(2024, 3, 10, 12, 30)
    assert when[4].replace(tzinfo=None) == datetime(2024, 12, 31, 23, 59, 59)
    assert df["wait"].to_list() == [timedelta(minutes=1), timedelta(minutes=2.5), None, timedelta(0), timedelta(hours=1)]

    assert rsm.data.read_description(out) == DESCR
    assert pl.read_parquet_metadata(out)["description"] == DESCR
    assert (typed_rds.parent / "typed_description.md").read_text() == DESCR
    printed = capsys.readouterr().out
    assert "typed.rds ->" in printed and "5 rows, 10 columns, with description" in printed
    assert "note: lst: list column stored as text" in printed


@needs_r
def test_rda_objects_globs_out_dir_and_overwrite(tmp_path):
    src = tmp_path / "r"
    src.mkdir()
    _r(f"""
    first <- data.frame(a = 1:3); attr(first, "description") <- "# First"
    second <- data.frame(b = c("x", "y"))
    note <- "not a data frame"
    save(first, second, note, file = "{src / 'both.rda'}")
    saveRDS(lm(dist ~ speed, data = cars), "{src / 'model.rds'}")
    saveRDS(data.frame(z = 1), "{src / 'plain.rds'}")
    """)
    (src / "readme.txt").write_text("ignored")
    out = tmp_path / "parquet"
    written = rsm.data.rds_to_parquet(str(src / "*"), out_dir=out, quiet=True)
    assert sorted(p.name for p in written) == ["first.parquet", "plain.parquet", "second.parquet"]
    assert rsm.data.read_description(out / "first.parquet") == "# First"
    assert rsm.data.read_description(out / "second.parquet") is None
    assert not (out / "second_description.md").exists()

    # a folder works like its files; existing parquet files are kept with overwrite=False
    assert rsm.data.rds_to_parquet(src, out_dir=out, overwrite=False, quiet=True) == []


@needs_r
def test_files_that_are_not_data_are_reported(tmp_path, capsys):
    _r(f'saveRDS(lm(dist ~ speed, data = cars), "{tmp_path / "model.rds"}")')
    (tmp_path / "broken.rds").write_bytes(b"not an rds file")
    assert rsm.data.rds_to_parquet([tmp_path / "model.rds", tmp_path / "broken.rds"]) == []
    printed = capsys.readouterr().out
    assert "model.rds: skipped, holds no data frame (lm)" in printed
    assert "broken.rds: not converted:" in printed


def test_paths_that_match_nothing(tmp_path):
    with pytest.raises(FileNotFoundError, match="No files match"):
        rsm.data.rds_to_parquet(str(tmp_path / "*.rds"))
    with pytest.raises(FileNotFoundError, match="no file or folder"):
        rsm.data.rds_to_parquet(tmp_path / "missing.rds")
    (tmp_path / "data.csv").write_text("a\n1\n")
    with pytest.raises(ValueError, match="not an R data file"):
        rsm.data.rds_to_parquet(tmp_path / "data.csv")


def test_a_clear_message_when_r_is_missing(tmp_path, monkeypatch, capsys):
    (tmp_path / "x.rds").write_bytes(b"")
    monkeypatch.delenv("RSCRIPT", raising=False)
    monkeypatch.setattr(rds_mod.shutil, "which", lambda *a, **k: None)
    monkeypatch.setattr(rds_mod, "_RSCRIPT_LOCATIONS", ())
    with pytest.raises(RuntimeError, match="R could not be found"):
        rsm.data.rds_to_parquet(tmp_path / "x.rds")
    with pytest.raises(RuntimeError, match="There is no Rscript at"):
        rds_mod.find_rscript("/nowhere/Rscript")
    assert rds_mod.main([str(tmp_path / "x.rds")]) == 1
    assert "rds2parquet: R could not be found" in capsys.readouterr().err


RADIANT_DIAMONDS = Path(__file__).parents[2] / "radiant.data" / "data" / "diamonds.rda"


@needs_r
@pytest.mark.skipif(not RADIANT_DIAMONDS.exists(), reason="radiant.data checkout not found")
def test_radiant_diamonds_matches_the_shipped_example(tmp_path):
    (out,) = rsm.data.rds_to_parquet(RADIANT_DIAMONDS, out_dir=tmp_path, quiet=True)
    shipped = Path(__file__).parents[1] / "examples" / "data" / "data" / "diamonds.parquet"
    assert pl.read_parquet(out).equals(pl.read_parquet(shipped))
    assert rsm.data.read_description(out).startswith("## Diamond prices")


def test_a_timeout_stops_r_with_a_clear_message(tmp_path, monkeypatch):
    import subprocess

    import pytest

    from pyrsm.data import rds as rds_module

    src = tmp_path / "x.rds"
    src.write_bytes(b"not really R data")

    def too_slow(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout"))

    monkeypatch.setattr(rds_module, "find_rscript", lambda rscript=None: "Rscript")
    monkeypatch.setattr(rds_module.subprocess, "run", too_slow)
    with pytest.raises(RuntimeError, match="within 5 seconds"):
        rds_module.rds_to_parquet(src, timeout=5, quiet=True)
