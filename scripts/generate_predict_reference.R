#!/usr/bin/env Rscript
# Generate prediction-parity fixtures: radiant.model::predict(...) for every
# model pyrsm ports (regress, logistic, mnl, nb, crtree, nn, rforest, gbt),
# under every prediction mode (pred_data, pred_cmd, data + pred_cmd) and the
# estimation options that change the prediction path.
#
# Writes one tests/reference/radiant_model/predict_<model>_<case>.json per
# case. Each holds the model call (args), the prediction call (pred_data /
# pred_cmd / data_cmd and options), the prediction table as R returns it (all
# columns; long tables keep their first and last 10 rows plus summary
# statistics of every numeric column) and the first lines of R's printout.
# Also writes the small synthetic datasets both languages read
# (predict_data_<name>.csv with their column types in predict_data_schema.json),
# a fingerprint of the package datasets (predict_data_fingerprint.json) so the
# tests can check pyrsm's parquet copies are the same data, and sessionInfo().
#
# Usage (from the pyrsm root):
#   OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 Rscript scripts/generate_predict_reference.R
#
# Tests: tests/test_predict_radiant_parity.py

suppressMessages({
  library(radiant.model)
  library(radiant.data)
  library(jsonlite)
})

OUT <- file.path("tests", "reference", "radiant_model")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

## remove fixtures from an earlier run so a dropped case does not linger
old <- list.files(OUT, pattern = "^predict_.*\\.json$", full.names = TRUE)
old <- old[!grepl("^predict_data_", basename(old))]
invisible(file.remove(old))

write_json_file <- function(obj, path) {
  writeLines(toJSON(obj, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), path)
}

# -- data -------------------------------------------------------------------

## package datasets (pyrsm has the same data as examples/data/model/*.parquet)
for (nm in c("diamonds", "titanic", "dvd", "ketchup", "houseprices")) {
  assign(nm, get(nm), envir = globalenv())
}
catalog <- radiant.model::catalog
names(catalog)[names(catalog) == "HH.size"] <- "HH_size" # the parquet's name
assign("catalog", catalog, envir = globalenv())

col_type <- function(x) {
  if (is.factor(x)) {
    list(enum = levels(x))
  } else if (is.logical(x)) {
    "bool"
  } else if (is.integer(x)) {
    "int"
  } else if (inherits(x, "Date")) {
    "date"
  } else if (is.numeric(x)) {
    "float"
  } else {
    "str"
  }
}

fingerprint <- function(d) {
  list(
    nrow = nrow(d),
    types = lapply(d, col_type),
    sums = lapply(Filter(function(x) is.numeric(x) && !inherits(x, "Date"), d), function(x) sum(as.numeric(x)))
  )
}
write_json_file(
  lapply(list(diamonds = diamonds, titanic = titanic, dvd = dvd, ketchup = ketchup,
              houseprices = houseprices, catalog = catalog), fingerprint),
  file.path(OUT, "predict_data_fingerprint.json")
)

## synthetic data, written to CSV and read back, so R and pyrsm use the same values
SCHEMA <- list()
write_synth <- function(name, d) {
  path <- file.path(OUT, paste0("predict_data_", name, ".csv"))
  write.csv(d, path, row.names = FALSE, na = "")
  SCHEMA[[name]] <<- lapply(d, col_type)
  back <- read.csv(path, stringsAsFactors = FALSE, na.strings = "")
  for (v in names(d)) {
    tp <- SCHEMA[[name]][[v]]
    back[[v]] <- if (is.list(tp)) {
      factor(back[[v]], levels = tp$enum)
    } else if (identical(tp, "int")) {
      as.integer(back[[v]])
    } else if (identical(tp, "float")) {
      as.numeric(back[[v]])
    } else if (identical(tp, "bool")) {
      as.logical(back[[v]])
    } else {
      as.character(back[[v]])
    }
  }
  stopifnot(isTRUE(all.equal(back, d, check.attributes = FALSE)))
  assign(name, back, envir = globalenv())
  cat("wrote", path, "\n")
}

set.seed(20261102)
n <- 120
x1 <- round(rnorm(n, 10, 3), 2)
xi <- sample(1:6, n, replace = TRUE)
ext <- round(exp(rnorm(n, 8, 3)), 2) # 0.1 .. 1e7: extreme values
flag <- sample(c(TRUE, FALSE), n, replace = TRUE, prob = c(0.4, 0.6))
flag50 <- c(TRUE, sample(rep(c(TRUE, FALSE), c(n / 2 - 1, n / 2)))) # an exact tie, TRUE first
dom <- factor(sample(c("rare1", "common", "rare2"), n, replace = TRUE, prob = c(0.1, 0.8, 0.1)),
              levels = c("rare1", "common", "rare2")) # the dominant level is not the first
## tied modes: "hi" and "mid" occur 45 times each. As a factor the level order
## (lo, hi, mid) breaks the tie; as text R's table() is alphabetical (hi);
## the first row is "mid", so first-appearance order would pick "mid"
tie_v <- c("mid", sample(rep(c("lo", "hi", "mid"), c(30, 45, 44))))
tie <- factor(tie_v, levels = c("lo", "hi", "mid"))
tie_s <- tie_v
xna <- round(x1 + rnorm(n, 0, 2), 2)
xna[sample(n, 14)] <- NA
cna <- factor(sample(c("p", "q", "r"), n, replace = TRUE, prob = c(0.3, 0.45, 0.25)), levels = c("p", "q", "r"))
cna[sample(n, 11)] <- NA
y <- round(5 + 0.8 * x1 + 0.5 * xi + 1.5 * flag + c(-1, 0, 2)[as.integer(dom)] +
             c(0, 1, -1)[as.integer(tie)] + 3e-7 * ext + rnorm(n, 0, 2), 3)
yna <- y
yna[sample(n, 16)] <- NA
yb <- factor(ifelse(y + rnorm(n, 0, 2) > median(y), "yes", "no"), levels = c("yes", "no"))
ym <- cut(y + rnorm(n, 0, 2), breaks = c(-Inf, quantile(y, c(0.33, 0.66)), Inf), labels = c("A", "B", "C"))
w <- sample(1:4, n, replace = TRUE)
write_synth("synth", data.frame(
  y = y, yna = yna, yb = yb, ym = ym, x1 = x1, xi = xi, ext = ext, flag = flag, flag50 = flag50,
  dom = dom, tie = tie, tie_s = tie_s, xna = xna, cna = cna, w = w, stringsAsFactors = FALSE
))

## frequency-weighted titanic (same estimates as the full data)
titanic_agg <- titanic %>%
  dplyr::count(pclass, sex, survived, name = "n") %>%
  as.data.frame()
titanic_agg$n <- as.integer(titanic_agg$n)
write_synth("titanic_agg", titanic_agg)

write_json_file(SCHEMA, file.path(OUT, "predict_data_schema.json"))

# -- serialization ----------------------------------------------------------

frame_spec <- function(d) {
  d <- as.data.frame(d, stringsAsFactors = FALSE)
  list(
    columns = colnames(d),
    types = lapply(d, col_type),
    values = lapply(d, function(x) if (is.factor(x) || inherits(x, "Date")) as.character(x) else x),
    nrow = nrow(d)
  )
}

## the prediction table: all columns; long tables keep rows 1-10 and the last
## 10 (row numbers in `rows`) plus summary statistics of numeric columns
table_spec <- function(pred) {
  d <- as.data.frame(pred, stringsAsFactors = FALSE)
  attributes(d) <- attributes(d)[c("names", "row.names", "class")]
  nr <- nrow(d)
  idx <- if (nr > 30) c(1:10, (nr - 9):nr) else seq_len(nr)
  out <- frame_spec(d[idx, , drop = FALSE])
  out$nrow <- nr
  out$rows <- idx
  out$truncated <- nr > 30
  num <- Filter(is.numeric, d)
  out$stats <- lapply(num, function(x) list(
    n = sum(!is.na(x)), mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE),
    min = min(x, na.rm = TRUE), max = max(x, na.rm = TRUE)
  ))
  out
}

## an R value (vector) as R code
r_values <- function(x) {
  s <- if (is.character(x)) {
    paste0("'", x, "'")
  } else if (is.logical(x)) {
    as.character(x)
  } else {
    vapply(x, function(v) format(v, digits = 15), "")
  }
  if (length(x) == 1) s else paste0("c(", paste(s, collapse = ", "), ")")
}

## a command list -> R's pred_cmd string (";" between variables for data + cmd)
r_cmd <- function(cmd, sep = ", ") {
  paste(vapply(names(cmd), function(v) paste0(v, " = ", r_values(cmd[[v]])), ""), collapse = sep)
}

FITS <- list()
fit_model <- function(model, args) {
  key <- paste(model, toJSON(args, auto_unbox = TRUE, digits = NA))
  if (is.null(FITS[[key]])) {
    fun <- get(model, envir = asNamespace("radiant.model"))
    res <- do.call(fun, c(args, list(envir = globalenv())))
    if (is.character(res)) stop(paste(model, ":", res))
    FITS[[key]] <<- res
  }
  FITS[[key]]
}

N_CASES <- 0
run_case <- function(id, model, args, data = NULL, cmd = NULL, opts = list(), note = "") {
  res <- fit_model(model, args)
  pred_args <- list(object = res)
  pred <- list()
  if (!is.null(data)) {
    if (is.character(data)) {
      pred_args$pred_data <- data
      pred$data <- list(dataset = data)
    } else {
      pred_args$pred_data <- data
      pred$data <- frame_spec(data)
    }
  }
  if (!is.null(cmd)) {
    pred$cmd <- lapply(cmd, I)
    pred_args$pred_cmd <- r_cmd(cmd, sep = if (is.null(data)) ", " else "; ")
    pred$r_cmd <- pred_args$pred_cmd
  }
  pred$mode <- if (is.null(data)) "cmd" else if (is.null(cmd)) "data" else "data_cmd"
  pred_args <- c(pred_args, opts, list(envir = globalenv()))
  out <- tryCatch(do.call(predict, pred_args), error = function(e) paste("R error:", conditionMessage(e)))
  fx <- list(case = id, model = model, args = args, pred = pred, opts = opts, note = note,
             r_class = class(res)[1], r_evar = res$evar)
  if (is.character(out)) {
    fx$error <- out
  } else {
    fx$table <- table_spec(out)
    fx$printed <- head(capture.output(print(out, n = 10)), 30)
  }
  path <- file.path(OUT, paste0("predict_", model, "_", id, ".json"))
  write_json_file(fx, path)
  N_CASES <<- N_CASES + 1
  invisible(fx)
}

## prediction data helpers
cols <- function(d, v, rows = 1:12) as.data.frame(d)[rows, v, drop = FALSE]
lev_subset <- function(d, var, keep, v, nr = 12) {
  d <- as.data.frame(d)
  d <- d[as.character(d[[var]]) %in% keep, v, drop = FALSE][1:nr, , drop = FALSE]
  d[[var]] <- factor(as.character(d[[var]]), levels = keep)
  d
}
as_text <- function(d, var) {
  d[[var]] <- as.character(d[[var]])
  d
}

# -- regress ----------------------------------------------------------------

dia <- list(dataset = "diamonds", rvar = "price", evar = c("carat", "clarity"))
dia4 <- list(dataset = "diamonds", rvar = "price", evar = c("carat", "clarity", "cut", "color"))
hp <- list(dataset = "houseprices", rvar = "price",
           evar = c("squarefeet", "bedrooms", "bathrooms", "brick", "neighborhood"))
cat_ <- list(dataset = "catalog", rvar = "Sales", evar = c("Income", "HH_size", "Age"))
syn <- list(dataset = "synth", rvar = "y", evar = c("x1", "xi", "flag", "dom", "tie"))
NOSE <- list(se = FALSE)

run_case("dia_data", "regress", dia, data = "diamonds", opts = NOSE)
run_case("dia_newcols", "regress", dia, data = cols(diamonds, c("clarity", "carat")), opts = NOSE,
         note = "only the model's columns, in a different order")
run_case("dia_levsub", "regress", dia, opts = NOSE,
         data = lev_subset(diamonds, "clarity", c("VS2", "SI1"), c("clarity", "carat")),
         note = "clarity has 2 of 8 levels, in a different order")
run_case("dia_text", "regress", dia, opts = NOSE,
         data = as_text(cols(diamonds, c("carat", "clarity")), "clarity"),
         note = "clarity as text in the new data")
run_case("dia_int_carat", "regress", dia, opts = NOSE,
         data = data.frame(carat = c(1L, 2L, 3L), clarity = factor(c("SI1", "VS1", "IF"), levels = levels(diamonds$clarity))),
         note = "carat (float in the estimation data) as an integer column")
run_case("dia_cmd_one", "regress", dia, cmd = list(carat = 1), opts = NOSE)
run_case("dia_cmd_three", "regress", dia, cmd = list(carat = c(0.5, 1, 2)), opts = NOSE)
run_case("dia_cmd_two_vars", "regress", dia, cmd = list(carat = c(0.5, 1), clarity = c("SI1", "VS1", "IF")),
         opts = NOSE, note = "grid of two variables: rows and their order")
run_case("dia_cmd_levels_text", "regress", dia, cmd = list(clarity = c("IF", "I1", "SI2")), opts = NOSE,
         note = "levels as strings, not in level order")
run_case("dia4_cmd_unmentioned", "regress", dia4, cmd = list(carat = c(0.5, 1)), opts = NOSE,
         note = "clarity, cut and color at their most frequent level")
run_case("dia4_cmd_unmentioned_cat", "regress", dia4, cmd = list(cut = c("Fair", "Ideal")), opts = NOSE,
         note = "carat at its mean, clarity and color at their most frequent level")
run_case("dia_datacmd", "regress", dia, data = cols(diamonds, c("carat", "clarity")), cmd = list(carat = 1), opts = NOSE)
run_case("dia4_datacmd_two", "regress", dia4, data = cols(diamonds, c("carat", "clarity", "cut", "color")),
         cmd = list(carat = 1.5, cut = "Ideal"), opts = NOSE)
run_case("dia_int_cmd", "regress", c(dia, list(int = "carat:clarity")), opts = NOSE,
         cmd = list(carat = c(0.5, 1, 2), clarity = c("SI2", "VVS1")))
run_case("dia_int_data", "regress", c(dia, list(int = "carat:clarity")), data = "diamonds", opts = NOSE)
run_case("dia_quad_cmd", "regress", list(dataset = "diamonds", rvar = "price", evar = "carat", int = "I(carat^2)"),
         cmd = list(carat = c(0.5, 1, 2, 3)), opts = NOSE)
run_case("dia_quad_data", "regress", list(dataset = "diamonds", rvar = "price", evar = c("carat", "clarity"), int = "I(carat^2)"),
         data = cols(diamonds, c("carat", "clarity"), 1:15), opts = NOSE)
run_case("dia_std_cmd", "regress", c(dia4, list(check = "standardize")), cmd = list(carat = c(0.5, 1, 2)), opts = NOSE)
run_case("dia_std_data", "regress", c(dia4, list(check = "standardize")), data = "diamonds", opts = NOSE)
run_case("dia_center_cmd", "regress", c(dia, list(check = "center")), cmd = list(clarity = c("SI1", "IF")), opts = NOSE)
run_case("dia_center_data", "regress", c(dia, list(check = "center")),
         data = cols(diamonds, c("carat", "clarity"), 1:15), opts = NOSE)
run_case("dia_filter_cmd", "regress", c(dia4, list(data_filter = "price > 5000")),
         cmd = list(clarity = c("SI1", "IF")), opts = NOSE,
         note = "carat mean and modes from the filtered rows")
run_case("dia_filter_data", "regress", c(dia, list(data_filter = "price > 5000")),
         data = cols(diamonds, c("carat", "clarity")), opts = NOSE,
         note = "new data are not filtered")
run_case("dia_arr_rows_cmd", "regress", c(dia4, list(arr = "desc(carat)", rows = 1:500)),
         cmd = list(clarity = c("SI1", "IF")), opts = NOSE,
         note = "carat mean and modes from the 500 largest diamonds")
run_case("hp_cmd_int_means", "regress", hp, cmd = list(brick = c("No", "Yes")), opts = NOSE,
         note = "squarefeet, bedrooms and bathrooms are integers: their means are not whole numbers")
run_case("hp_cmd_int_values", "regress", hp, cmd = list(squarefeet = c(1500.5, 2250.25)), opts = NOSE,
         note = "non-integer values for an integer column")
run_case("hp_data", "regress", hp, data = "houseprices", opts = NOSE)
run_case("hp_float_data", "regress", hp, opts = NOSE,
         data = data.frame(squarefeet = c(1750.5, 2100, 2600.25), bedrooms = c(2.5, 3, 4),
                           bathrooms = c(2, 2.5, 3), brick = c("No", "Yes", "No"),
                           neighborhood = factor(c("East", "West", "North"), levels = c("East", "North", "West"))),
         note = "integer columns as floats (with fractions), brick as text")
run_case("catalog_cmd", "regress", cat_, cmd = list(Income = c(20, 50, 80)), opts = NOSE,
         note = "HH_size is an integer column")
run_case("catalog_data", "regress", cat_, data = "catalog", opts = NOSE)

## intervals
run_case("dia_ci95_cmd", "regress", dia, cmd = list(carat = c(0.5, 1, 2)),
         opts = list(se = TRUE, interval = "confidence", conf_lev = 0.95))
run_case("dia_ci90_cmd", "regress", dia, cmd = list(carat = c(0.5, 1, 2)),
         opts = list(se = TRUE, interval = "confidence", conf_lev = 0.90))
run_case("dia_ci99_data", "regress", dia, data = cols(diamonds, c("carat", "clarity"), 1:15),
         opts = list(se = TRUE, interval = "confidence", conf_lev = 0.99))
run_case("dia_pi95_cmd", "regress", dia, cmd = list(carat = c(0.5, 1, 2)),
         opts = list(se = TRUE, interval = "prediction", conf_lev = 0.95))
run_case("dia_pi80_data", "regress", dia, data = cols(diamonds, c("carat", "clarity"), 1:15),
         opts = list(se = TRUE, interval = "prediction", conf_lev = 0.80))
run_case("hp_pi90_cmd", "regress", hp, cmd = list(neighborhood = c("East", "North", "West")),
         opts = list(se = TRUE, interval = "prediction", conf_lev = 0.90))
run_case("dia_std_ci_cmd", "regress", c(dia, list(check = "standardize")), cmd = list(carat = c(0.5, 1)),
         opts = list(se = TRUE, interval = "confidence", conf_lev = 0.95),
         note = "radiant.model gives no interval for a standardized model")

## synthetic edge cases
run_case("syn_cmd_fill", "regress", syn, cmd = list(x1 = c(8, 12)), opts = NOSE,
         note = "xi (int) at its mean, flag (bool) by majority, dom at its dominant level, tie at the first tied level")
run_case("syn_cmd_bool", "regress", syn, cmd = list(flag = c(TRUE, FALSE)), opts = NOSE)
run_case("syn_data", "regress", syn, data = "synth", opts = NOSE)
run_case("syn_tie_text", "regress", list(dataset = "synth", rvar = "y", evar = c("x1", "tie_s")),
         cmd = list(x1 = 10), opts = NOSE, note = "tied modes in a text column")
run_case("syn_flag50", "regress", list(dataset = "synth", rvar = "y", evar = c("x1", "flag50")),
         cmd = list(x1 = 10), opts = NOSE, note = "a bool with exactly half TRUE")
run_case("syn_xi_frac_cmd", "regress", syn, cmd = list(xi = c(1.5, 2.5)), opts = NOSE,
         note = "fractions for an integer column")
run_case("syn_ext_cmd", "regress", list(dataset = "synth", rvar = "y", evar = c("x1", "ext")),
         cmd = list(ext = c(0.01, 1e4, 1e7)), opts = NOSE)
run_case("syn_ext_data", "regress", list(dataset = "synth", rvar = "y", evar = c("x1", "ext")),
         data = "synth", opts = NOSE)
run_case("syn_na_x_cmd", "regress", list(dataset = "synth", rvar = "y", evar = c("x1", "xna", "cna")),
         cmd = list(x1 = c(8, 12)), opts = NOSE,
         note = "xna and cna have missing values: means/modes from the complete rows the model used")
run_case("syn_na_y_cmd", "regress", list(dataset = "synth", rvar = "yna", evar = c("x1", "xi", "dom")),
         cmd = list(dom = c("rare1", "rare2")), opts = NOSE,
         note = "yna has missing values: x1/xi means from the rows the model used")
run_case("syn_na_data", "regress", list(dataset = "synth", rvar = "y", evar = c("x1", "xna", "cna")),
         data = "synth", opts = NOSE, note = "rows with a missing explanatory value are dropped")
run_case("syn_newlevel", "regress", syn, opts = NOSE,
         data = data.frame(x1 = c(9, 11), xi = c(2L, 3L), flag = c(TRUE, FALSE), dom = c("common", "zz"),
                           tie = factor(c("lo", "hi"), levels = c("lo", "hi", "mid")), stringsAsFactors = FALSE),
         note = "a level not seen in estimation: an error")
run_case("syn_filter_cmd", "regress", c(syn, list(data_filter = "x1 > 9")), cmd = list(flag = c(TRUE, FALSE)), opts = NOSE)

# -- logistic ---------------------------------------------------------------

tit <- list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex", "age"), lev = "Yes")
tit_int <- list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex", "sibsp", "parch"), lev = "Yes")
dvd_ <- list(dataset = "dvd", rvar = "buy", evar = c("coupon", "purch", "last"), lev = "yes")
NOSE <- list(se = FALSE)

run_case("tit_data", "logistic", tit, data = "titanic", opts = NOSE)
run_case("tit_newcols", "logistic", tit, data = cols(titanic, c("age", "sex", "pclass")), opts = NOSE)
run_case("tit_levsub", "logistic", tit, opts = NOSE,
         data = lev_subset(titanic, "pclass", c("3rd", "1st"), c("age", "sex", "pclass")))
run_case("tit_text", "logistic", tit, data = as_text(as_text(cols(titanic, c("pclass", "sex", "age")), "pclass"), "sex"), opts = NOSE)
run_case("tit_cmd_one", "logistic", tit, cmd = list(age = 30), opts = NOSE)
run_case("tit_cmd_levels", "logistic", tit, cmd = list(pclass = c("1st", "2nd", "3rd")), opts = NOSE)
run_case("tit_cmd_two_vars", "logistic", tit, cmd = list(age = c(10, 30, 60), sex = c("female", "male")), opts = NOSE)
run_case("tit_cmd_levels_text", "logistic", tit, cmd = list(pclass = c("3rd", "1st")), opts = NOSE)
run_case("tit_datacmd", "logistic", tit, data = cols(titanic, c("pclass", "sex", "age"), 1:10), cmd = list(sex = "male"), opts = NOSE)
run_case("tit_int_cmd", "logistic", c(tit, list(int = "pclass:sex")), cmd = list(pclass = c("1st", "3rd"), sex = c("female", "male")), opts = NOSE)
run_case("tit_quad_cmd", "logistic", c(tit, list(int = "I(age^2)")), cmd = list(age = c(5, 25, 45, 65)), opts = NOSE)
run_case("tit_std_cmd", "logistic", c(tit, list(check = "standardize")), cmd = list(age = c(10, 50)), opts = NOSE)
run_case("tit_std_data", "logistic", c(tit, list(check = "standardize")), data = "titanic", opts = NOSE)
run_case("tit_center_cmd", "logistic", c(tit, list(check = "center")), cmd = list(sex = c("female", "male")), opts = NOSE)
run_case("tit_lev_no", "logistic", modifyList(tit, list(lev = "No")), cmd = list(pclass = c("1st", "3rd")), opts = NOSE)
run_case("tit_wts_cmd", "logistic",
         list(dataset = "titanic_agg", rvar = "survived", evar = c("pclass", "sex"), lev = "Yes", wts = "n"),
         cmd = list(pclass = c("1st", "2nd", "3rd")), opts = NOSE,
         note = "frequency weights: the most frequent sex is counted by rows of the aggregated data")
run_case("tit_wts_data", "logistic",
         list(dataset = "titanic_agg", rvar = "survived", evar = c("pclass", "sex"), lev = "Yes", wts = "n"),
         data = "titanic_agg", opts = NOSE)
run_case("tit_filter_cmd", "logistic", c(tit, list(data_filter = "age > 10")), cmd = list(sex = c("female", "male")), opts = NOSE)
run_case("tit_arr_rows_cmd", "logistic", c(tit, list(arr = "desc(fare)", rows = 1:500)),
         cmd = list(sex = c("female", "male")), opts = NOSE)
run_case("tit_int_cols_cmd", "logistic", tit_int, cmd = list(sex = c("female", "male")), opts = NOSE,
         note = "sibsp and parch are integers: their means are not whole numbers")
run_case("dvd_cmd", "logistic", dvd_, cmd = list(coupon = c(0, 1)), opts = NOSE,
         note = "purch and last are integers")
run_case("dvd_data", "logistic", dvd_, data = "dvd", opts = NOSE)
run_case("dvd_datacmd_frac", "logistic", dvd_, data = cols(dvd, c("coupon", "purch", "last"), 1:10),
         cmd = list(purch = 2.5), opts = NOSE, note = "a fraction for an integer column")
run_case("syn_newlevel", "logistic", list(dataset = "synth", rvar = "yb", evar = c("x1", "dom"), lev = "yes"), opts = NOSE,
         data = data.frame(x1 = c(9, 11), dom = c("common", "zz"), stringsAsFactors = FALSE))
run_case("tit_ci95_cmd", "logistic", tit, cmd = list(age = c(10, 30, 60)), opts = list(se = TRUE, conf_lev = 0.95))
run_case("tit_ci90_data", "logistic", tit, data = cols(titanic, c("pclass", "sex", "age"), 1:15), opts = list(se = TRUE, conf_lev = 0.90))
run_case("syn_cmd_fill", "logistic", list(dataset = "synth", rvar = "yb", evar = c("x1", "xi", "flag", "dom", "tie"), lev = "yes"),
         cmd = list(x1 = c(8, 12)), opts = NOSE)
run_case("syn_na_x_cmd", "logistic", list(dataset = "synth", rvar = "yb", evar = c("x1", "xna", "cna"), lev = "yes"),
         cmd = list(x1 = c(8, 12)), opts = NOSE)
run_case("syn_na_data", "logistic", list(dataset = "synth", rvar = "yb", evar = c("x1", "xna", "cna"), lev = "yes"),
         data = "synth", opts = NOSE)
run_case("syn_wts_cmd", "logistic", list(dataset = "synth", rvar = "yb", evar = c("x1", "xi", "dom"), lev = "yes", wts = "w"),
         cmd = list(dom = c("rare1", "common")), opts = NOSE)

# -- mnl --------------------------------------------------------------------

prices <- c("price.heinz28", "price.heinz32", "price.heinz41", "price.hunts32")
ket <- list(dataset = "ketchup", rvar = "choice", evar = prices, lev = "heinz28")
ket_d <- list(dataset = "ketchup", rvar = "choice", evar = c("price.heinz28", "price.heinz32", "disp.heinz28", "feat.heinz28"), lev = "heinz28")

run_case("ket_data", "mnl", ket, data = "ketchup")
run_case("ket_newcols", "mnl", ket, data = cols(ketchup, rev(prices)))
run_case("ket_cmd_one", "mnl", ket, cmd = list(price.heinz28 = 4))
run_case("ket_cmd_three", "mnl", ket, cmd = list(price.heinz28 = c(3.5, 4.3, 5.2)))
run_case("ket_cmd_two_vars", "mnl", ket_d, cmd = list(price.heinz28 = c(3.5, 5), disp.heinz28 = c("no", "yes")))
run_case("ket_cmd_unmentioned", "mnl", ket_d, cmd = list(price.heinz32 = c(3, 3.5)),
         note = "disp/feat at their most frequent level")
run_case("ket_text", "mnl", ket_d, data = as_text(cols(ketchup, rev(ket_d$evar)), "disp.heinz28"))
run_case("ket_levsub", "mnl", ket_d, data = lev_subset(ketchup, "disp.heinz28", c("yes"), ket_d$evar))
run_case("ket_datacmd", "mnl", ket_d, data = cols(ketchup, ket_d$evar), cmd = list(disp.heinz28 = "yes"))
run_case("ket_base_hunts", "mnl", modifyList(ket, list(lev = "hunts32")), cmd = list(price.hunts32 = c(3, 4)))
run_case("ket_int", "mnl", c(ket_d, list(int = "price.heinz28:disp.heinz28")),
         cmd = list(price.heinz28 = c(3.5, 5), disp.heinz28 = c("no", "yes")))
run_case("ket_std_cmd", "mnl", c(ket, list(check = "standardize")), cmd = list(price.heinz28 = c(3.5, 5)))
run_case("ket_center_data", "mnl", c(ket, list(check = "center")), data = cols(ketchup, prices, 1:15))
run_case("ket_filter_cmd", "mnl", c(ket, list(data_filter = "price.heinz28 > 4")), cmd = list(price.heinz32 = c(3, 3.5)))
run_case("tit_wts_cmd", "mnl", list(dataset = "titanic_agg", rvar = "pclass", evar = c("sex", "survived"), lev = "1st", wts = "n"),
         cmd = list(sex = c("female", "male")))
run_case("tit_cmd_age", "mnl", list(dataset = "titanic", rvar = "pclass", evar = c("sex", "age", "sibsp"), lev = "1st"),
         cmd = list(age = c(20, 40)), note = "sibsp is an integer column")
run_case("syn_cmd_fill", "mnl", list(dataset = "synth", rvar = "ym", evar = c("x1", "xi", "dom"), lev = "A"),
         cmd = list(x1 = c(8, 12)))
run_case("syn_newlevel", "mnl", list(dataset = "synth", rvar = "ym", evar = c("x1", "dom"), lev = "A"),
         data = data.frame(x1 = c(9, 11), dom = c("common", "zz"), stringsAsFactors = FALSE))

# -- nb ---------------------------------------------------------------------

tnb <- list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex", "age"))
run_case("tit_newcols", "nb", tnb, data = cols(titanic, c("age", "sex", "pclass")))
run_case("tit_levsub", "nb", tnb, data = lev_subset(titanic, "pclass", c("3rd", "1st"), c("age", "sex", "pclass")))
run_case("tit_text", "nb", tnb, data = as_text(cols(titanic, c("pclass", "sex", "age")), "sex"))
run_case("tit_cmd_one", "nb", tnb, cmd = list(age = 30))
run_case("tit_cmd_two_vars", "nb", tnb, cmd = list(age = c(10, 30, 60), sex = c("female", "male")))
run_case("tit_cmd_levels_text", "nb", tnb, cmd = list(pclass = c("3rd", "1st")))
run_case("tit_datacmd", "nb", tnb, data = cols(titanic, c("pclass", "sex", "age"), 1:10), cmd = list(sex = "male"))
run_case("tit_filter_cmd", "nb", c(tnb, list(data_filter = "age > 10")), cmd = list(sex = c("female", "male")))
run_case("tit_arr_rows_cmd", "nb", c(tnb, list(arr = "desc(fare)", rows = 1:500)), cmd = list(sex = c("female", "male")))
run_case("tit_int_cols_cmd", "nb", list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sibsp", "parch")),
         cmd = list(pclass = c("1st", "3rd")))
run_case("dvd_cmd", "nb", list(dataset = "dvd", rvar = "buy", evar = c("coupon", "purch", "last")),
         cmd = list(coupon = c(0, 1)))
run_case("dvd_datacmd_frac", "nb", list(dataset = "dvd", rvar = "buy", evar = c("coupon", "purch", "last")),
         data = cols(dvd, c("coupon", "purch", "last"), 1:10), cmd = list(purch = 2.5),
         note = "a fraction for an integer column")
run_case("syn_newlevel", "nb", list(dataset = "synth", rvar = "yb", evar = c("x1", "dom")),
         data = data.frame(x1 = c(9, 11), dom = c("common", "zz"), stringsAsFactors = FALSE))
run_case("syn_cmd_fill", "nb", list(dataset = "synth", rvar = "ym", evar = c("x1", "xi", "flag", "dom", "tie")),
         cmd = list(x1 = c(8, 12)))
run_case("syn_tie_text", "nb", list(dataset = "synth", rvar = "yb", evar = c("x1", "tie_s")), cmd = list(x1 = 10))
run_case("syn_na_x_cmd", "nb", list(dataset = "synth", rvar = "yb", evar = c("x1", "xna", "cna")), cmd = list(x1 = c(8, 12)))
run_case("syn_na_data", "nb", list(dataset = "synth", rvar = "yb", evar = c("x1", "xna", "cna")), data = "synth")
run_case("syn_bool_data", "nb", list(dataset = "synth", rvar = "yb", evar = c("x1", "flag", "dom")), data = "synth")

# -- crtree -----------------------------------------------------------------

tct <- list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex", "age"), lev = "Yes")
dct <- list(dataset = "diamonds", rvar = "price", evar = c("carat", "clarity", "cut"), type = "regression", cp = 0.005)

run_case("tit_data", "crtree", tct, data = "titanic")
run_case("tit_newcols", "crtree", tct, data = cols(titanic, c("age", "sex", "pclass")))
run_case("tit_levsub", "crtree", tct, data = lev_subset(titanic, "pclass", c("3rd", "1st"), c("age", "sex", "pclass")))
run_case("tit_text", "crtree", tct, data = as_text(cols(titanic, c("pclass", "sex", "age")), "pclass"))
run_case("tit_cmd_one", "crtree", tct, cmd = list(age = 30))
run_case("tit_cmd_two_vars", "crtree", tct, cmd = list(age = c(5, 30, 60), pclass = c("1st", "3rd")))
run_case("tit_cmd_levels_text", "crtree", tct, cmd = list(pclass = c("3rd", "1st")))
run_case("tit_datacmd", "crtree", tct, data = cols(titanic, c("pclass", "sex", "age"), 1:10), cmd = list(sex = "female"))
run_case("tit_lev_no", "crtree", modifyList(tct, list(lev = "No")), cmd = list(pclass = c("1st", "2nd", "3rd")))
run_case("tit_pcp", "crtree", c(tct, list(pcp = 0.01)), cmd = list(age = c(5, 30, 60)))
run_case("tit_nodes", "crtree", c(tct, list(nodes = 4)), data = cols(titanic, c("pclass", "sex", "age"), 1:15))
run_case("tit_cp", "crtree", modifyList(tct, list(cp = 0.02)), data = "titanic")
run_case("tit_prior", "crtree", c(tct, list(prior = 0.5)), cmd = list(pclass = c("1st", "2nd", "3rd")))
run_case("tit_prior_noadj", "crtree", c(tct, list(prior = 0.5, adjprob = FALSE)), cmd = list(pclass = c("1st", "2nd", "3rd")))
run_case("tit_cost", "crtree", c(tct, list(cost = 1, margin = 5)), data = cols(titanic, c("pclass", "sex", "age"), 1:15))
run_case("tit_wts", "crtree", list(dataset = "titanic_agg", rvar = "survived", evar = c("pclass", "sex"), lev = "Yes", wts = "n"),
         cmd = list(pclass = c("1st", "2nd", "3rd")))
run_case("tit_filter_cmd", "crtree", c(tct, list(data_filter = "age > 10")), cmd = list(pclass = c("1st", "3rd")))
run_case("tit_arr_rows_cmd", "crtree", c(tct, list(arr = "desc(fare)", rows = 1:500)), cmd = list(pclass = c("1st", "3rd")))
run_case("tit_std_cmd", "crtree", c(tct, list(check = "standardize")), cmd = list(age = c(5, 30, 60)))
run_case("dvd_cmd", "crtree", list(dataset = "dvd", rvar = "buy", evar = c("coupon", "purch", "last"), lev = "yes", cp = 0.002),
         cmd = list(coupon = c(0, 1)), note = "purch and last are integers")
run_case("dia_data", "crtree", dct, data = "diamonds")
run_case("dvd_datacmd_frac", "crtree", list(dataset = "dvd", rvar = "buy", evar = c("coupon", "purch", "last"), lev = "yes", cp = 0.002),
         data = cols(dvd, c("coupon", "purch", "last"), 1:10), cmd = list(purch = 2.5),
         note = "a fraction for an integer column")
run_case("dia_cmd", "crtree", dct, cmd = list(carat = c(0.5, 1, 2)))
run_case("dia_cmd_two_vars", "crtree", dct, cmd = list(carat = c(0.5, 1.5), clarity = c("SI2", "IF")))
run_case("dia_nodes", "crtree", c(dct, list(nodes = 6)), data = cols(diamonds, c("carat", "clarity", "cut"), 1:15))
run_case("syn_na_data", "crtree", list(dataset = "synth", rvar = "yb", evar = c("x1", "xna", "cna"), lev = "yes"),
         data = "synth", note = "rows with a missing explanatory value are dropped")
run_case("syn_newlevel", "crtree", list(dataset = "synth", rvar = "yb", evar = c("x1", "dom"), lev = "yes"),
         data = data.frame(x1 = c(6, 9, 14), dom = c("common", "zz", "rare2"), stringsAsFactors = FALSE),
         note = "rpart sends an unseen level down the majority path")
run_case("syn_cmd_fill", "crtree", list(dataset = "synth", rvar = "y", evar = c("x1", "xi", "flag", "dom", "tie"), type = "regression"),
         cmd = list(x1 = c(8, 12)))

# -- nn (nnet; pyrsm's mlp is sklearn: compare the grid, record the gap) ---

tnn <- list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex", "age"), lev = "Yes", size = 2, decay = 0.5, seed = 1234)
dnn <- list(dataset = "diamonds", rvar = "price", evar = c("carat", "clarity"), type = "regression", size = 2, decay = 0.5, seed = 1234)
run_case("tit_data", "nn", tnn, data = "titanic")
run_case("tit_newcols", "nn", tnn, data = cols(titanic, c("age", "sex", "pclass")))
run_case("tit_levsub", "nn", tnn, data = lev_subset(titanic, "pclass", c("3rd", "1st"), c("age", "sex", "pclass")))
run_case("tit_text", "nn", tnn, data = as_text(cols(titanic, c("pclass", "sex", "age")), "sex"))
run_case("tit_cmd_two_vars", "nn", tnn, cmd = list(age = c(10, 30, 60), sex = c("female", "male")))
run_case("tit_datacmd", "nn", tnn, data = cols(titanic, c("pclass", "sex", "age"), 1:10), cmd = list(sex = "male"))
run_case("dvd_cmd", "nn", list(dataset = "dvd", rvar = "buy", evar = c("coupon", "purch", "last"), lev = "yes", size = 2, decay = 0.5, seed = 1234),
         cmd = list(coupon = c(0, 1)), note = "purch and last are integers")
run_case("dia_data", "nn", dnn, data = "diamonds")
run_case("dia_cmd", "nn", dnn, cmd = list(carat = c(0.5, 1, 2)))

# -- rforest (ranger; pyrsm is sklearn: compare the grid, record the gap) --

trf <- list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex", "age"), lev = "Yes", seed = 1234, num.threads = 1)
drf <- list(dataset = "diamonds", rvar = "price", evar = c("carat", "clarity"), type = "regression", seed = 1234, num.threads = 1)
run_case("tit_data", "rforest", trf, data = "titanic", note = "R uses OOB predictions when the data are the estimation data")
run_case("tit_newcols", "rforest", trf, data = cols(titanic, c("age", "sex", "pclass")))
run_case("tit_levsub", "rforest", trf, data = lev_subset(titanic, "pclass", c("3rd", "1st"), c("age", "sex", "pclass")))
run_case("tit_text", "rforest", trf, data = as_text(cols(titanic, c("pclass", "sex", "age")), "sex"))
run_case("tit_cmd_two_vars", "rforest", trf, cmd = list(age = c(10, 30, 60), sex = c("female", "male")))
run_case("tit_datacmd", "rforest", trf, data = cols(titanic, c("pclass", "sex", "age"), 1:10), cmd = list(sex = "male"))
run_case("dvd_cmd", "rforest", list(dataset = "dvd", rvar = "buy", evar = c("coupon", "purch", "last"), lev = "yes", seed = 1234, num.threads = 1),
         cmd = list(coupon = c(0, 1)), note = "purch and last are integers")
run_case("dia_data", "rforest", drf, data = "diamonds", note = "R uses OOB predictions when the data are the estimation data")
run_case("dia_cmd", "rforest", drf, cmd = list(carat = c(0.5, 1, 2)))
run_case("dia_newcols", "rforest", drf, data = cols(diamonds, c("clarity", "carat")))

# -- gbt (xgboost) ----------------------------------------------------------

xgb_opts <- list(nthread = 1, seed = 1234)
dvd_gbt <- c(list(dataset = "dvd", rvar = "buy", evar = c("coupon", "purch", "last"), lev = "yes"), xgb_opts)
cat_gbt <- c(list(dataset = "catalog", rvar = "Sales", evar = c("Income", "HH_size", "Age"), type = "regression"), xgb_opts)
tit_gbt <- c(list(dataset = "titanic", rvar = "survived", evar = c("sex", "age", "fare"), lev = "Yes"), xgb_opts)
tit3_gbt <- c(list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex", "age"), lev = "Yes"), xgb_opts)
dia_gbt <- c(list(dataset = "diamonds", rvar = "price", evar = c("carat", "clarity"), type = "regression"), xgb_opts)
run_case("dvd_data", "gbt", dvd_gbt, data = "dvd", note = "numeric explanatory variables only")
run_case("dvd_cmd", "gbt", dvd_gbt, cmd = list(coupon = c(0, 1)), note = "purch and last are integers")
run_case("catalog_data", "gbt", cat_gbt, data = "catalog")
run_case("catalog_cmd", "gbt", cat_gbt, cmd = list(Income = c(20, 50, 80)))
run_case("tit_bin_data", "gbt", tit_gbt, data = "titanic", note = "a two-level factor and numbers")
run_case("tit_bin_newcols", "gbt", tit_gbt, data = cols(titanic, c("fare", "age", "sex")))
run_case("tit_bin_text", "gbt", tit_gbt, data = as_text(cols(titanic, c("sex", "age", "fare")), "sex"))
run_case("tit_bin_cmd", "gbt", tit_gbt, cmd = list(sex = c("female", "male"), age = c(10, 40)))
run_case("tit_data", "gbt", tit3_gbt, data = "titanic", note = "pclass has three levels")
run_case("tit_cmd", "gbt", tit3_gbt, cmd = list(pclass = c("1st", "2nd", "3rd")))
run_case("tit_levsub", "gbt", tit3_gbt, data = lev_subset(titanic, "pclass", c("3rd", "1st"), c("age", "sex", "pclass")))
run_case("tit_datacmd", "gbt", tit3_gbt, data = cols(titanic, c("pclass", "sex", "age"), 1:10), cmd = list(sex = "male"))
run_case("dia_data", "gbt", dia_gbt, data = "diamonds")
run_case("dia_cmd", "gbt", dia_gbt, cmd = list(carat = c(0.5, 1, 2)))

writeLines(capture.output(sessionInfo()), file.path(OUT, "predict_sessionInfo.txt"))
cat("wrote", N_CASES, "prediction fixtures\n")
