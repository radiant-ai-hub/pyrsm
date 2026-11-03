#!/usr/bin/env Rscript
# Generate estimation-parity fixtures: the full estimation output of
# radiant.model::regress(...) and radiant.model::logistic(...) and their
# summary(...) options, for pyrsm's rsm.model.regress and rsm.model.logistic.
#
# Writes one tests/reference/radiant_model/est_<model>_<case>.json per case.
# Each holds the radiant.model call (args), R's coefficient table (raw term
# names and radiant's labels, estimate, std.error, t/z, p.value; for logistic
# the odds ratio), the aliased terms, the fit statistics radiant prints, the
# coefficient confidence intervals (radiant's choice: confint / profile /
# confint.default / confint_robust), car::vif, the variables test (test_var),
# the sum of squares and RMSE (regress), the stepwise path (step()$anova),
# residuals and fitted values (first and last 10 plus summary statistics),
# warnings, and the first lines of R's printed summary.
#
# Logistic: R's glm() takes the covariance from the IRLS weights at the start
# of its last iteration, i.e. at the coefficients before the last update
# (default epsilon = 1e-8), so radiant's standard errors are up to 5e-4 off the
# converged ones (titanic with pclass:sex). Every logistic fixture therefore
# also holds a `tight` block: the same model refit with glm.control(epsilon =
# 1e-14, maxit = 100) (and the default QR rank tolerance, see glm_fit_tol), with
# its standard errors (classical or HC1), Wald / robust intervals,
# profile-likelihood intervals found exactly (uniroot on the profile deviance;
# R's confint interpolates a spline through a grid of profile points, good to
# ~1e-3 standard errors), car::vif and the test.
#
# Also writes the synthetic edge-case data both languages read
# (est_data_<name>.csv, column types in est_data_schema.json), a fingerprint
# of the package datasets (est_data_fingerprint.json) so the tests can check
# pyrsm's parquet copies are the same data, and sessionInfo().
#
# Usage (from the pyrsm root):
#   OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 Rscript scripts/generate_estimation_reference.R
#
# Tests: tests/test_estimation_radiant_parity.py

suppressMessages({
  library(radiant.model)
  library(radiant.data)
  library(jsonlite)
})

OUT <- file.path("tests", "reference", "radiant_model")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

## remove fixtures from an earlier run so a dropped case does not linger
old <- list.files(OUT, pattern = "^est_(regress|logistic)_.*\\.json$", full.names = TRUE)
invisible(file.remove(old))

write_json_file <- function(obj, path) {
  writeLines(toJSON(obj, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), path)
}

# -- data -------------------------------------------------------------------

## package datasets (pyrsm has the same data as examples/data/model/*.parquet)
for (nm in c("diamonds", "titanic", "dvd", "houseprices", "ideal", "direct_marketing")) {
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
  lapply(list(diamonds = diamonds, titanic = titanic, dvd = dvd, houseprices = houseprices,
              ideal = ideal, catalog = catalog, direct_marketing = direct_marketing), fingerprint),
  file.path(OUT, "est_data_fingerprint.json")
)

## synthetic data, written to CSV and read back, so R and pyrsm use the same values
SCHEMA <- list()
write_synth <- function(name, d) {
  path <- file.path(OUT, paste0("est_data_", name, ".csv"))
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
  same <- mapply(function(a, b) isTRUE(all.equal(a, b, check.attributes = FALSE, tolerance = 0)), back, d)
  if (!all(same)) cat("  round trip not exact (15 significant digits) for:", names(d)[!same], "\n")
  stopifnot(isTRUE(all.equal(back, d, check.attributes = FALSE)))
  assign(name, back, envir = globalenv())
  cat("wrote", path, "\n")
}

set.seed(20261103)
n <- 200
x1 <- round(rnorm(n, 10, 3), 2)
x2 <- round(rnorm(n, 0, 1), 3)
xi <- sample(1:6, n, replace = TRUE)
ext <- round(exp(rnorm(n, 8, 3)), 2) # 0.1 .. 1e7: extreme values
flag <- sample(c(TRUE, FALSE), n, replace = TRUE, prob = c(0.4, 0.6))
## a factor whose levels are not alphabetical; the base level is rare
dom <- factor(sample(c("rare1", "common", "rare2"), n, replace = TRUE, prob = c(0.1, 0.8, 0.1)),
              levels = c("rare1", "common", "rare2"))
## the same groups as text (R: alphabetical levels, base "alpha") and as a
## factor in a non-alphabetical order (base "delta"); the first row is "delta"
grp_s <- c("delta", sample(c("delta", "alpha", "charlie", "bravo"), n - 1, replace = TRUE))
grp_f <- factor(grp_s, levels = c("delta", "alpha", "charlie", "bravo"))
## a rare level: 3 of 200 rows
rare <- sample(c("a", "b"), n, replace = TRUE)
rare[c(17, 88, 140)] <- "tiny"
rare <- factor(rare, levels = c("a", "b", "tiny"))
## perfectly collinear columns
xcol <- 2 * x1 + 3
xsum <- x1 + x2
is_alpha <- as.numeric(grp_s == "alpha") # the grp alpha dummy
const <- rep(5, n)
## missing values
xna <- round(x1 + rnorm(n, 0, 2), 2)
xna[sample(n, 15)] <- NA
cna <- factor(sample(c("p", "q", "r"), n, replace = TRUE, prob = c(0.3, 0.45, 0.25)), levels = c("p", "q", "r"))
cna[sample(n, 11)] <- NA
y <- round(3 + 0.8 * x1 - 1.2 * x2 + 0.3 * xi + 1.5 * flag + c(-1, 0, 2)[as.integer(dom)] +
             c(0, 1, -1, 0.5)[as.integer(grp_f)] + 3e-7 * ext + rnorm(n, 0, 2), 3)
yna <- y
yna[sample(n, 12)] <- NA
yi <- as.integer(round(y)) # an integer response
eta <- -0.3 + 0.35 * (x1 - 10) - 0.9 * x2 + 0.7 * flag + c(0.5, 0, -0.8)[as.integer(dom)] +
  c(0, 0.6, -0.4, 0.2)[as.integer(grp_f)]
yb <- factor(ifelse(runif(n) < plogis(eta), "yes", "no"), levels = c("yes", "no"))
yb_rev <- factor(as.character(yb), levels = c("no", "yes")) # first level "no"
yb_s <- as.character(yb) # text response
## a text response whose first value ("reject") is not the alphabetically first
yb_t <- ifelse(yb == "yes", "accept", "reject")
y01 <- as.integer(yb == "yes") # integer 0/1 response
ylog <- yb == "yes" # logical response
ybna <- yb
ybna[sample(n, 10)] <- NA
## quasi-separation: every row of level "pure" is a "yes"
qsep <- sample(c("a", "b"), n, replace = TRUE)
qsep[which(yb == "yes")[1:8]] <- "pure"
qsep <- factor(qsep, levels = c("a", "b", "pure"))
w <- sample(1:4, n, replace = TRUE) # integer frequency weights
wf <- round(runif(n, 0.5, 3), 2) # non-integer weights, many values
ws <- sample(c(0.5, 2), n, replace = TRUE) # sampling weights: radiant turns on robust
write_synth("synth", data.frame(
  y = y, yna = yna, yi = yi, yb = yb, yb_rev = yb_rev, yb_s = yb_s, yb_t = yb_t, y01 = y01, ylog = ylog, ybna = ybna,
  x1 = x1, x2 = x2, xi = xi, ext = ext, flag = flag, dom = dom, grp_s = grp_s, grp_f = grp_f,
  rare = rare, xcol = xcol, xsum = xsum, is_alpha = is_alpha, const = const, xna = xna, cna = cna,
  qsep = qsep, w = w, wf = wf, ws = ws, stringsAsFactors = FALSE
))

## perfect separation: y is "yes" exactly when x > 0
set.seed(20261104)
ns <- 60
sx <- round(rnorm(ns), 3)
write_synth("sep", data.frame(
  y = factor(ifelse(sx > 0, "yes", "no"), levels = c("yes", "no")), x = sx,
  z = round(rnorm(ns), 3), stringsAsFactors = FALSE
))

## frequency-weighted titanic (same estimates as the full data)
titanic_agg <- titanic %>%
  dplyr::count(pclass, sex, survived, name = "n") %>%
  as.data.frame()
titanic_agg$n <- as.integer(titanic_agg$n)
write_synth("titanic_agg", titanic_agg)

## titanic with integer weights and a noise column (for stepwise)
titanic_w <- as.data.frame(titanic)[, c("survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked")]
titanic_w$noise <- round(sin(seq_len(nrow(titanic_w))), 6)
titanic_w$w <- as.integer(1 + seq_len(nrow(titanic_w)) %% 3)
write_synth("titanic_w", titanic_w)

write_json_file(SCHEMA, file.path(OUT, "est_data_schema.json"))

# -- serialization helpers --------------------------------------------------

## a vector of residuals / fitted values: all of it, or the first and last 10
## (positions `idx` in the model's rows, `data_rows` in the scoped data)
## plus summary statistics of the whole vector
vec_spec <- function(x) {
  n <- length(x)
  idx <- if (n > 30) c(1:10, (n - 9):n) else seq_len(n)
  nm <- names(x)
  list(
    n = n, idx = idx,
    data_rows = if (is.null(nm)) NULL else as.integer(nm[idx]),
    values = unname(x[idx]),
    stats = list(mean = mean(x), sd = sd(x), min = min(x), max = max(x), sum_sq = sum(x^2))
  )
}

ci_spec <- function(cnf, level, type) {
  cnf <- as.matrix(cnf)
  list(level = level, type = type, term = rownames(cnf), low = unname(cnf[, 1]), high = unname(cnf[, 2]))
}

## run f, collecting warnings (radiant.model wraps glm() in sshhr(), so its
## own call shows none) and printed output
quietly <- function(expr) {
  warns <- character(0)
  out <- NULL
  printed <- capture.output({
    out <- withCallingHandlers(
      tryCatch(expr, error = function(e) structure(conditionMessage(e), class = "r_error")),
      warning = function(w) {
        warns <<- c(warns, conditionMessage(w))
        invokeRestart("muffleWarning")
      },
      message = function(m) invokeRestart("muffleMessage")
    )
  })
  list(value = out, warnings = unique(warns), printed = printed)
}

summary_lines <- function(res, sum_check, conf_lev, test_var) {
  out <- quietly(summary(res, sum_check = sum_check, conf_lev = conf_lev, test_var = test_var))
  head(out$printed, 80)
}

## radiant.model's matchCf (summary.regress / summary.logistic, test_var)
matchCf <- function(clist, vlist) {
  matcher <- function(vl, cn) {
    if (grepl(":", vl)) {
      strsplit(vl, ":") %>% unlist() %>% sapply(function(x) gsub("var", x, "((var.*:)|(:var))")) %>%
        paste0(collapse = "|") %>% grepl(cn) %>% cn[.]
    } else {
      mf <- grepl(paste0("^", vl, "$"), cn) %>% cn[.]
      if (length(mf) == 0) mf <- grepl(paste0("^", vl), cn) %>% cn[.]
      mf
    }
  }
  cn <- names(clist)
  sapply(vlist, matcher, cn) %>% unname()
}

## the test variables and the restricted model, as summary.regress/logistic
test_setup <- function(object, test_var) {
  vars <- object$evar
  if (!is.empty(object$int) && length(vars) > 1) {
    capture.output(test_var <- unique(c(test_var, test_specs(test_var, object$int))))
    vars <- c(vars, object$int)
  }
  list(test_var = test_var, not_selected = base::setdiff(vars, test_var))
}

N_CASES <- 0

# -- regress ----------------------------------------------------------------

## summary.regress's test_var block, returning its numbers
regress_test <- function(object, test_var) {
  if (any(grepl("stepwise", object$check))) return(list(skipped = "stepwise"))
  ts <- test_setup(object, test_var)
  sub_form <- paste(object$rvar, "~ 1")
  if (length(ts$not_selected) > 0) sub_form <- paste(". ~", paste(ts$not_selected, collapse = " + "))
  sub_mod <- update(object$model, sub_form, data = object$model$model) %>% anova(object$model, test = "F")
  out <- list(test_var = ts$test_var, sub_form = sub_form)
  if (is.na(sub_mod[, "Pr(>F)"][2])) {
    out$no_test <- TRUE
    return(out)
  }
  out$heading <- attr(sub_mod, "heading")[2]
  tss <- object$model$model[, 1] %>% (function(x) sum((x - mean(x))^2))
  if ("robust" %in% object$check) {
    sub_mod <- tryCatch(
      car::linearHypothesis(object$model, matchCf(object$model$coef, ts$test_var), vcov = object$vcov),
      error = function(e) conditionMessage(e)
    )
    if (is.character(sub_mod)) {
      out$error <- sub_mod
      return(out)
    }
  }
  r2 <- 1 - (sub_mod$RSS / tss)
  out$r2 <- if (length(r2) == 2) r2 else NULL # NULL: R prints no R-squared (robust)
  out$stat <- sub_mod$F[2]
  out$df1 <- sub_mod$Res.Df[1] - sub_mod$Res.Df[2]
  out$df2 <- sub_mod$Res.Df[2]
  out$p_value <- sub_mod[, "Pr(>F)"][2]
  out
}

reg_case <- function(id, args, sum_check = c("rmse", "sumsquares", "vif", "confint"), test_var = "",
                     conf_lev = 0.95, py = list(), note = "") {
  fit <- quietly(do.call(regress, c(args, list(envir = globalenv()))))
  res <- fit$value
  fx <- list(case = id, model = "regress", args = args, py = py, sum_check = sum_check,
             test_var = test_var, conf_lev = conf_lev, note = note, warnings = fit$warnings)
  if (inherits(res, "r_error") || is.character(res)) {
    fx$error <- as.character(res)
  } else {
    m <- res$model
    cf <- coef(m)
    fx$r_evar <- res$evar
    fx$int <- res$int
    fx$check <- res$check
    fx$terms_raw <- names(cf)
    fx$aliased <- names(cf)[is.na(cf)]
    co <- res$coeff
    td <- broom::tidy(m) %>% na.omit()
    stopifnot(nrow(td) == nrow(co))
    fx$coef <- list(term = td$term, label = co$label, estimate = co$coefficient, std_error = co$std.error,
                    statistic = co$t.value, p_value = co$p.value)
    gl <- broom::glance(m)
    atab <- anova(m)
    nr <- nrow(atab)
    fx$fit <- list(
      r2 = gl$r.squared, adj_r2 = gl$adj.r.squared, fstat = unname(gl$statistic), df_model = gl$df,
      df_resid = gl$df.residual, p_value = unname(gl$p.value), nobs = gl$nobs, sigma = gl$sigma,
      rmse = sqrt(mean(m$residuals^2, na.rm = TRUE)),
      ss_reg = sum(atab$`Sum Sq`[-nr]), ss_err = atab$`Sum Sq`[nr], ss_df_reg = sum(atab$Df[-nr]),
      ss_df_err = atab$Df[nr]
    )
    if (!anyNA(cf)) {
      cnf <- if ("robust" %in% res$check) confint_robust(m, level = conf_lev, dist = "t") else confint(m, level = conf_lev)
      fx$ci <- ci_spec(cnf, conf_lev, if ("robust" %in% res$check) "robust_t" else "t")
      if (length(attr(m$terms, "term.labels")) > 1) {
        v <- quietly(car::vif(m))
        gv <- if (is.null(dim(v$value))) v$value else v$value[, "GVIF"]
        fx$vif <- list(term = names(gv), vif = unname(gv), warnings = v$warnings)
      }
    }
    if (!is.empty(test_var)) {
      fx$test <- regress_test(res, test_var)
    }
    if (!is.null(m$anova)) {
      fx$step <- list(anova = lapply(as.list(m$anova), function(x) if (is.factor(x)) as.character(x) else x),
                      terms = attr(terms(m), "term.labels"))
    }
    fx$resid <- vec_spec(m$residuals)
    fx$fitted_values <- vec_spec(m$fitted.values)
    fx$printed <- summary_lines(res, sum_check, conf_lev, test_var)
  }
  write_json_file(fx, file.path(OUT, paste0("est_regress_", id, ".json")))
  N_CASES <<- N_CASES + 1
  invisible(fx)
}

dia <- list(dataset = "diamonds", rvar = "price", evar = c("carat", "clarity"))
dia4 <- list(dataset = "diamonds", rvar = "price", evar = c("carat", "clarity", "cut", "color"))
dia_all <- list(dataset = "diamonds", rvar = "price",
                evar = c("carat", "clarity", "cut", "color", "depth", "table", "x", "y", "z"))
hp <- list(dataset = "houseprices", rvar = "price",
           evar = c("squarefeet", "bedrooms", "bathrooms", "brick", "neighborhood"))
syn <- list(dataset = "synth", rvar = "y", evar = c("x1", "x2", "xi", "flag", "dom", "grp_f"))

reg_case("dia", dia, test_var = "clarity")
reg_case("dia4", dia4, test_var = c("cut", "color"))
reg_case("dia_carat_only", list(dataset = "diamonds", rvar = "price", evar = "carat"), test_var = "carat",
         note = "one explanatory variable: no VIF; the test is against the intercept-only model")
reg_case("dia_int_num_cat", c(dia, list(int = "carat:clarity")), test_var = "carat",
         note = "the test of carat includes carat:clarity (test_specs)")
reg_case("dia_int_cat_cat", list(dataset = "diamonds", rvar = "price", evar = c("carat", "cut", "color"), int = "cut:color"),
         test_var = "cut")
reg_case("dia_quad", list(dataset = "diamonds", rvar = "price", evar = c("carat", "clarity"), int = "I(carat^2)"),
         test_var = "I(carat^2)")
reg_case("dia_quad_test_carat", list(dataset = "diamonds", rvar = "price", evar = c("carat", "clarity"), int = "I(carat^2)"),
         test_var = "carat", note = "test_specs leaves I(carat^2) in the restricted model")
reg_case("dia_std", c(dia4, list(check = "standardize")), test_var = "color")
reg_case("dia_std1", c(dia4, list(check = "standardize-1sd")))
reg_case("dia_center", c(dia4, list(check = "center")))
reg_case("dia_std_int", c(dia, list(int = "carat:clarity", check = "standardize")))
reg_case("dia_robust", c(dia4, list(check = "robust")), test_var = "cut")
reg_case("dia_robust_std", c(dia4, list(check = c("robust", "standardize"))), test_var = "clarity")
reg_case("dia_robust_int", c(dia, list(int = "carat:clarity", check = "robust")), test_var = "carat")
reg_case("dia_filter", c(dia4, list(data_filter = "price > 5000")), test_var = "cut")
reg_case("dia_filter_base", c(dia, list(data_filter = "clarity != 'I1'")), test_var = "clarity",
         py = list(data_filter = "clarity != 'I1'"),
         note = "the filter removes the base level I1: SI2 becomes the base")
reg_case("dia_arr_rows", c(dia4, list(arr = "desc(carat)", rows = 1:500)), test_var = "color")
reg_case("dia_quad_only", list(dataset = "diamonds", rvar = "price", evar = "carat", int = "I(carat^2)"),
         note = "one variable and its square: two terms, so radiant shows the VIF")
reg_case("dia_date", list(dataset = "diamonds", rvar = "price", evar = c("carat", "date")),
         note = "a Date column enters as days since 1970-01-01")
reg_case("dia_step_back", c(dia_all, list(check = "stepwise-backward")))
reg_case("dia_step_fwd", c(dia_all, list(check = "stepwise-forward")))
reg_case("dia_step_both", c(dia_all, list(check = "stepwise-both")))
reg_case("dia_step_int", list(dataset = "diamonds", rvar = "price", evar = c("carat", "clarity", "depth", "table"),
                              int = c("carat:clarity", "depth:table"), check = "stepwise-backward"))
reg_case("hp", hp, test_var = "neighborhood")
reg_case("hp_robust", c(hp, list(check = "robust")), test_var = c("bedrooms", "bathrooms"))
reg_case("hp_step_both", c(hp, list(check = "stepwise-both")))
reg_case("catalog", list(dataset = "catalog", rvar = "Sales", evar = c("Income", "HH_size", "Age")), test_var = "Age")
reg_case("catalog_conf90", list(dataset = "catalog", rvar = "Sales", evar = c("Income", "HH_size", "Age")),
         conf_lev = 0.90)
reg_case("ideal", list(dataset = "ideal", rvar = "y", evar = c("x1", "x2", "x3")), test_var = c("x2", "x3"))
reg_case("dm", list(dataset = "direct_marketing", rvar = "AmountSpent",
                    evar = c("Age", "Gender", "Salary", "Children", "History", "Catalogs")), test_var = "History")
reg_case("dm_all_test", list(dataset = "direct_marketing", rvar = "AmountSpent", evar = c("Salary", "Catalogs")),
         test_var = c("Salary", "Catalogs"), note = "every variable tested: the restricted model is the intercept")

## synthetic edge cases
reg_case("syn", syn, test_var = "grp_f")
reg_case("syn_text", list(dataset = "synth", rvar = "y", evar = c("x1", "grp_s")), test_var = "grp_s",
         note = "a text column: R's levels are alphabetical (base alpha)")
reg_case("syn_factor_order", list(dataset = "synth", rvar = "y", evar = c("x1", "grp_f")),
         note = "the same groups as a factor whose first level is delta (the base)")
reg_case("syn_bool", list(dataset = "synth", rvar = "y", evar = c("x1", "flag")), test_var = "flag")
reg_case("syn_rare", list(dataset = "synth", rvar = "y", evar = c("x1", "rare")), test_var = "rare",
         note = "level tiny has 3 rows")
reg_case("syn_dom", list(dataset = "synth", rvar = "y", evar = c("x1", "dom")),
         note = "the base level rare1 is rare; levels not alphabetical")
reg_case("syn_ext", list(dataset = "synth", rvar = "y", evar = c("x1", "ext")),
         note = "ext spans 0.1 .. 1e7")
reg_case("syn_int_resp", list(dataset = "synth", rvar = "yi", evar = c("x1", "x2", "xi")),
         note = "an integer response")
reg_case("syn_na_x", list(dataset = "synth", rvar = "y", evar = c("x1", "xna", "cna")), test_var = "cna",
         note = "missing values in explanatory variables: rows dropped")
reg_case("syn_na_y", list(dataset = "synth", rvar = "yna", evar = c("x1", "x2", "dom")), test_var = "dom",
         note = "missing values in the response: rows dropped")
reg_case("syn_na_std", list(dataset = "synth", rvar = "yna", evar = c("x1", "xna"), check = "standardize"),
         note = "get_data(na.rm = TRUE) drops incomplete rows first: means and sds of the estimation rows")
reg_case("syn_na_robust", list(dataset = "synth", rvar = "y", evar = c("x1", "xna"), check = "robust"),
         note = "robust standard errors with rows dropped for a missing value")
reg_case("syn_collinear", list(dataset = "synth", rvar = "y", evar = c("x1", "x2", "xcol")),
         note = "xcol = 2 x1 + 3: aliased, R drops xcol")
reg_case("syn_collinear_sum", list(dataset = "synth", rvar = "y", evar = c("xsum", "x1", "x2")), test_var = "x2",
         note = "xsum = x1 + x2: R drops x2 (the last)")
reg_case("syn_collinear_dummy", list(dataset = "synth", rvar = "y", evar = c("x1", "grp_f", "is_alpha")),
         note = "is_alpha equals the grp_f alpha dummy: R drops is_alpha")
reg_case("syn_const", list(dataset = "synth", rvar = "y", evar = c("x1", "const")),
         note = "a constant column: radiant refuses the model")

# -- logistic ---------------------------------------------------------------

## radiant's model-fit numbers (summary.logistic)
logit_fit <- function(m, object) {
  gl <- broom::glance(m)
  rnk <- m$rank
  llnull <- gl$null.deviance / -2
  llfull <- gl$deviance / -2
  nobs <- gl$nobs
  bic <- gl$BIC
  wts <- object$wts
  if (!is.empty(wts, "None") && (length(unique(wts)) > 2 || min(wts) >= 1)) {
    nobs <- sum(wts)
    bic <- -2 * gl$logLik + log(nobs) * (1 + gl$df.null - gl$df.residual)
  }
  list(
    r2 = 1 - llfull / llnull, r2_adj = 1 - (llfull - (rnk - 1)) / llnull,
    auc = radiant.model::auc(m$fitted.values, m$model[[object$rvar]]),
    loglik = gl$logLik, aic = gl$AIC, bic = bic, chi2 = gl$null.deviance - gl$deviance,
    df = gl$df.null - gl$df.residual,
    p_value = pchisq(gl$null.deviance - gl$deviance, gl$df.null - gl$df.residual, lower.tail = FALSE),
    nobs = nobs, deviance = gl$deviance, null_deviance = gl$null.deviance, glance_nobs = gl$nobs,
    glance_bic = gl$BIC
  )
}

## summary.logistic's test_var block, for model m (radiant's or the tight refit)
logistic_test <- function(object, test_var, m = object$model, vcov = object$vcov,
                          control = glm.control(), method = "glm.fit") {
  if (any(grepl("stepwise", object$check))) return(list(skipped = "stepwise"))
  ts <- test_setup(object, test_var)
  sub_form <- paste(object$rvar, "~ 1")
  if (length(ts$not_selected) > 0) sub_form <- paste(object$rvar, "~", paste(ts$not_selected, collapse = " + "))
  wts <- object$wts
  logit_sub <- tryCatch(
    sshhr(glm(as.formula(sub_form), weights = wts, family = binomial(link = "logit"), data = m$model,
              control = control, method = method)),
    error = function(e) conditionMessage(e)
  )
  out <- list(test_var = ts$test_var, sub_form = sub_form)
  if (is.character(logit_sub)) {
    out$error <- logit_sub
    return(out)
  }
  sub_test <- anova(logit_sub, m, test = "Chi")
  out$heading <- attr(sub_test, "heading")[2]
  out$r2 <- c((logit_sub$null.deviance - logit_sub$deviance) / logit_sub$null.deviance,
              1 - m$deviance / m$null.deviance)
  if ("robust" %in% object$check) {
    lh <- tryCatch(car::linearHypothesis(m, matchCf(m$coef, ts$test_var), vcov = vcov),
                   error = function(e) conditionMessage(e))
    if (is.character(lh)) {
      out$error <- lh
      return(out)
    }
    out$stat <- lh[2, "Chisq"]
    out$df <- lh[2, "Df"]
    out$p_value <- lh[2, "Pr(>Chisq)"]
  } else {
    out$stat <- sub_test[2, "Deviance"]
    out$df <- sub_test[2, "Df"]
    out$p_value <- sub_test[2, "Pr(>Chi)"]
  }
  out
}

TIGHT <- glm.control(epsilon = 1e-14, maxit = 100)

## glm.fit with the QR rank tolerance of the default control: glm.fit uses
## min(1e-7, epsilon / 1000), which at epsilon = 1e-14 is 1e-17 and would no
## longer find aliased columns (default and pyrsm: 1e-11)
glm_fit_tol <- glm.fit
local({
  src <- paste(deparse(body(glm.fit), width.cutoff = 500L), collapse = "\n")
  stopifnot(grepl("min(1e-07, control$epsilon/1000)", src, fixed = TRUE))
  body(glm_fit_tol) <<- parse(text = gsub("min(1e-07, control$epsilon/1000)", "1e-11", src, fixed = TRUE))[[1]]
})

## radiant's glm refit with a tight convergence criterion
tight_glm <- function(m) {
  d <- m$model
  if ("(weights)" %in% names(d)) {
    glm(formula(m), family = binomial(link = "logit"), data = d, weights = `(weights)`, control = TIGHT,
        method = glm_fit_tol)
  } else {
    glm(formula(m), family = binomial(link = "logit"), data = d, control = TIGHT, method = glm_fit_tol)
  }
}

## profile-likelihood interval found exactly: the coefficient values where the
## deviance of the model refit with that coefficient fixed exceeds the minimum
## by qchisq(level, 1)
profile_exact <- function(m, level) {
  X <- model.matrix(m)
  y <- m$y
  w <- m$prior.weights
  dev_hat <- m$deviance
  q <- qchisq(level, 1)
  b <- coef(m)
  se <- sqrt(diag(vcov(m)))
  out <- matrix(NA_real_, length(b), 2, dimnames = list(names(b), c("low", "high")))
  for (j in seq_along(b)) {
    excess <- function(bj) {
      if (ncol(X) == 1) {
        mu <- plogis(X[, j] * bj)
        dev <- sum(binomial()$dev.resids(y, mu, w))
      } else {
        dev <- suppressWarnings(glm_fit_tol(X[, -j, drop = FALSE], y, weights = w, offset = X[, j] * bj,
                                        family = binomial(), control = TIGHT))$deviance
      }
      dev - dev_hat - q
    }
    for (s in c(-1, 1)) {
      step <- se[j] * sqrt(q)
      edge <- b[j] + s * step
      k <- 0
      while (excess(edge) < 0 && k < 60) {
        step <- 2 * step
        edge <- b[j] + s * step
        k <- k + 1
      }
      ## a profile that never reaches the cutoff (separation): no bound, as
      ## confint's NA
      if (excess(edge) >= 0 && excess(b[j]) < 0) {
        out[j, (s + 3) / 2] <- uniroot(excess, sort(c(b[j], edge)), tol = 1e-13)$root
      }
    }
  }
  out
}

coef_block <- function(m, vc) {
  cf <- coef(m)
  ok <- !is.na(cf)
  se <- sqrt(diag(as.matrix(vc)))[names(cf)[ok]]
  z <- cf[ok] / se
  list(term = names(cf)[ok], estimate = unname(cf[ok]), std_error = unname(se), statistic = unname(z),
       p_value = unname(2 * pnorm(-abs(z))))
}

log_case <- function(id, args, sum_check = c("vif", "confint", "odds"), test_var = "", conf_lev = 0.95,
                     py = list(), note = "") {
  fit <- quietly(do.call(logistic, c(args, list(envir = globalenv()))))
  res <- fit$value
  fx <- list(case = id, model = "logistic", args = args, py = py, sum_check = sum_check,
             test_var = test_var, conf_lev = conf_lev, note = note, warnings = fit$warnings)
  if (inherits(res, "r_error") || is.character(res)) {
    fx$error <- as.character(res)
  } else {
    m <- res$model
    cf <- coef(m)
    robust <- "robust" %in% res$check
    fx$r_evar <- res$evar
    fx$int <- res$int
    fx$check <- res$check
    fx$lev <- res$lev
    fx$ci_type <- res$ci_type
    fx$robust <- robust
    fx$converged <- m$converged
    fx$iter <- m$iter
    ## the warnings radiant's sshhr() hides
    fx$glm_warnings <- quietly(
      if ("(weights)" %in% names(m$model)) {
        glm(formula(m), family = binomial(link = "logit"), data = m$model, weights = `(weights)`)
      } else {
        glm(formula(m), family = binomial(link = "logit"), data = m$model)
      }
    )$warnings
    fx$terms_raw <- names(cf)
    fx$aliased <- names(cf)[is.na(cf)]
    co <- res$coeff
    td <- broom::tidy(m) %>% na.omit()
    stopifnot(nrow(td) == nrow(co))
    fx$coef <- list(term = td$term, label = co$label, estimate = co$coefficient, std_error = co$std.error,
                    statistic = co$z.value, p_value = co$p.value, OR = co$OR, OR_pct = co$`OR%`)
    fx$fit <- logit_fit(m, res)
    if (!anyNA(cf)) {
      cnf <- if (robust) {
        confint_robust(m, level = conf_lev, vcov = res$vcov)
      } else if (res$ci_type == "profile") {
        suppressMessages(confint(m, level = conf_lev))
      } else {
        confint.default(m, level = conf_lev)
      }
      fx$ci <- ci_spec(cnf, conf_lev, if (robust) "robust" else res$ci_type)
      if (length(attr(m$terms, "term.labels")) > 1) {
        v <- quietly(car::vif(m))
        if (!inherits(v$value, "r_error")) {
          gv <- if (is.null(dim(v$value))) v$value else v$value[, "GVIF"]
          fx$vif <- list(term = names(gv), vif = unname(gv), warnings = v$warnings)
        }
      }
    }
    if (!is.empty(test_var)) fx$test <- logistic_test(res, test_var)
    if (!is.null(m$anova)) {
      fx$step <- list(anova = lapply(as.list(m$anova), function(x) if (is.factor(x)) as.character(x) else x),
                      terms = attr(terms(m), "term.labels"))
    }
    fx$resid <- vec_spec(m$residuals)
    fx$fitted_values <- vec_spec(m$fitted.values)

    ## the same model at a tight convergence criterion
    mt <- quietly(tight_glm(m))
    tw <- mt$warnings
    mt <- mt$value
    tcf <- coef(mt)
    vc <- if (robust) sandwich::vcovHC(mt, type = "HC1") else vcov(mt)
    tight <- c(coef_block(mt, vc), list(converged = mt$converged, iter = mt$iter, warnings = tw))
    tight$fit <- logit_fit(mt, res)
    if (!anyNA(tcf)) {
      tight$ci_wald <- ci_spec(if (robust) confint_robust(mt, level = conf_lev, vcov = vc) else confint.default(mt, level = conf_lev),
                               conf_lev, if (robust) "robust" else "default")
      if (!robust && res$ci_type == "profile") {
        tight$ci_profile <- ci_spec(profile_exact(mt, conf_lev), conf_lev, "profile_exact")
      }
      if (length(attr(mt$terms, "term.labels")) > 1) {
        v <- quietly(car::vif(mt))
        if (!inherits(v$value, "r_error")) {
          gv <- if (is.null(dim(v$value))) v$value else v$value[, "GVIF"]
          tight$vif <- list(term = names(gv), vif = unname(gv))
        }
      }
    }
    if (!is.empty(test_var)) tight$test <- logistic_test(res, test_var, m = mt, vcov = vc, control = TIGHT, method = glm_fit_tol)
    tight$resid <- vec_spec(mt$residuals)
    tight$fitted_values <- vec_spec(mt$fitted.values)
    fx$tight <- tight
    fx$printed <- summary_lines(res, sum_check, conf_lev, test_var)
  }
  write_json_file(fx, file.path(OUT, paste0("est_logistic_", id, ".json")))
  N_CASES <<- N_CASES + 1
  invisible(fx)
}

tit <- list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex", "age"), lev = "Yes")
titw_all <- list(dataset = "titanic_w", rvar = "survived",
                 evar = c("pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "noise"), lev = "Yes")
dvd_ <- list(dataset = "dvd", rvar = "buy", evar = c("coupon", "purch", "last"), lev = "yes")
lsyn <- list(dataset = "synth", rvar = "yb", evar = c("x1", "x2", "flag", "dom", "grp_f"), lev = "yes")

log_case("tit", tit, test_var = "pclass")
log_case("tit_lev_no", modifyList(tit, list(lev = "No")), test_var = "sex",
         note = "the second level as the level to predict")
log_case("tit_lev_default", modifyList(tit, list(lev = "")),
         note = "no level given: radiant uses the first factor level (Yes)")
log_case("tit_int", c(tit, list(int = "pclass:sex")), test_var = "sex",
         note = "the test of sex includes pclass:sex (test_specs)")
log_case("tit_int_num", c(tit, list(int = "sex:age")), test_var = "age")
log_case("tit_quad", c(tit, list(int = "I(age^2)")), test_var = "I(age^2)")
log_case("tit_quad_only", list(dataset = "titanic", rvar = "survived", evar = "age", int = "I(age^2)", lev = "Yes"),
         note = "one variable and its square: two terms, so radiant shows the VIF")
log_case("tit_std", c(tit, list(check = "standardize")), test_var = "age")
log_case("tit_std1", c(tit, list(check = "standardize-1sd")))
log_case("tit_center", c(tit, list(check = "center")))
log_case("tit_robust", c(tit, list(check = "robust")), test_var = "age")
log_case("tit_robust_int", c(tit, list(int = "pclass:sex", check = "robust")), test_var = "sex")
log_case("tit_robust_std", c(tit, list(check = c("robust", "standardize-1sd"))), test_var = "pclass")
log_case("tit_wald", c(tit, list(ci_type = "default")), py = list(ci_type = "wald"),
         note = "Wald intervals asked for (confint.default)")
log_case("tit_conf90", tit, conf_lev = 0.90)
log_case("tit_filter", c(tit, list(data_filter = "age > 10")), test_var = "age")
log_case("tit_arr_rows", c(tit, list(arr = "desc(fare)", rows = 1:500)), test_var = "sex")
log_case("tit_all", list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"),
                         lev = "Yes"), test_var = c("sibsp", "parch"))
log_case("tit_step_back", c(titw_all, list(check = "stepwise-backward")))
log_case("tit_step_fwd", c(titw_all, list(check = "stepwise-forward")))
log_case("tit_step_both", c(titw_all, list(check = "stepwise-both")))
log_case("tit_wts_agg", list(dataset = "titanic_agg", rvar = "survived", evar = c("pclass", "sex"), lev = "Yes", wts = "n"),
         test_var = "sex", note = "frequency weights: n per pclass x sex x survived")
log_case("tit_wts_int", c(list(dataset = "titanic_w", wts = "w"), tit[c("rvar", "evar", "lev")]), test_var = "age",
         note = "integer weights 1-3 per row")
log_case("tit_wts_std", c(list(dataset = "titanic_w", wts = "w", check = "standardize"), tit[c("rvar", "evar", "lev")]),
         note = "weighted means and sds for standardizing")
log_case("dvd", dvd_, test_var = "coupon", note = "20,000 rows: Wald intervals by default")
log_case("dvd_profile", c(dvd_, list(ci_type = "profile")), py = list(ci_type = "profile"))
log_case("dvd_robust", c(dvd_, list(check = "robust")), test_var = c("purch", "last"))

## synthetic edge cases
log_case("syn", lsyn, test_var = "grp_f")
log_case("syn_text_x", list(dataset = "synth", rvar = "yb", evar = c("x1", "grp_s"), lev = "yes"), test_var = "grp_s",
         note = "a text explanatory variable: R's levels are alphabetical (base alpha)")
log_case("syn_factor_order", list(dataset = "synth", rvar = "yb", evar = c("x1", "grp_f"), lev = "yes"),
         note = "the same groups as a factor whose first level is delta (the base)")
log_case("syn_resp_factor_default", list(dataset = "synth", rvar = "yb", evar = c("x1", "x2"), lev = ""),
         note = "factor response, no level: radiant uses the first level (yes)")
log_case("syn_resp_factor_rev_default", list(dataset = "synth", rvar = "yb_rev", evar = c("x1", "x2"), lev = ""),
         note = "factor response whose first level is no, no level given: radiant models no")
log_case("syn_resp_text", list(dataset = "synth", rvar = "yb_s", evar = c("x1", "x2"), lev = "yes"),
         note = "a text response")
log_case("syn_resp_text_default", list(dataset = "synth", rvar = "yb_s", evar = c("x1", "x2"), lev = ""),
         note = "a text response, no level: radiant uses the alphabetically first value (no)")
log_case("syn_resp_text2_default", list(dataset = "synth", rvar = "yb_t", evar = c("x1", "x2"), lev = ""),
         note = "a text response, no level: radiant uses the alphabetically first value (accept); the first row is reject")
log_case("syn_resp_01", list(dataset = "synth", rvar = "y01", evar = c("x1", "x2"), lev = "1"),
         note = "an integer 0/1 response, level 1")
log_case("syn_resp_01_default", list(dataset = "synth", rvar = "y01", evar = c("x1", "x2"), lev = ""),
         note = "an integer 0/1 response, no level: radiant uses the first sorted value, 0")
log_case("syn_resp_bool", list(dataset = "synth", rvar = "ylog", evar = c("x1", "x2"), lev = "TRUE"),
         note = "a logical response")
log_case("syn_bool_x", list(dataset = "synth", rvar = "yb", evar = c("x1", "flag"), lev = "yes"), test_var = "flag")
log_case("syn_rare", list(dataset = "synth", rvar = "yb", evar = c("x1", "rare"), lev = "yes"), test_var = "rare",
         note = "level tiny has 3 rows")
log_case("syn_dom", list(dataset = "synth", rvar = "yb", evar = c("x1", "x2", "dom"), lev = "yes"),
         note = "the base level rare1 is rare; levels not alphabetical")
log_case("syn_ext", list(dataset = "synth", rvar = "yb", evar = c("x1", "ext"), lev = "yes"),
         note = "ext spans 0.1 .. 1e7")
log_case("syn_na_x", list(dataset = "synth", rvar = "yb", evar = c("x1", "xna", "cna"), lev = "yes"), test_var = "cna",
         note = "missing values in explanatory variables: rows dropped")
log_case("syn_na_y", list(dataset = "synth", rvar = "ybna", evar = c("x1", "x2"), lev = "yes"), test_var = "x2",
         note = "missing values in the response: rows dropped")
log_case("syn_na_std", list(dataset = "synth", rvar = "yb", evar = c("x1", "xna"), lev = "yes", check = "standardize"),
         note = "get_data(na.rm = TRUE) drops incomplete rows first: means and sds of the estimation rows")
log_case("syn_wts_na", list(dataset = "synth", rvar = "yb", evar = c("x1", "xna"), lev = "yes", wts = "w"),
         note = "weights with rows dropped for a missing xna: Nr obs is the sum of the weights of the rows used")
log_case("syn_wts_frac", list(dataset = "synth", rvar = "yb", evar = c("x1", "x2"), lev = "yes", wts = "wf"),
         note = "non-integer weights with many values (not sampling weights)")
log_case("syn_wts_sampling", list(dataset = "synth", rvar = "yb", evar = c("x1", "x2"), lev = "yes", wts = "ws"),
         note = "two weight values, the smallest below 1: radiant turns on robust standard errors")
log_case("syn_collinear", list(dataset = "synth", rvar = "yb", evar = c("x1", "x2", "xcol"), lev = "yes"),
         note = "xcol = 2 x1 + 3: aliased, R drops xcol")
log_case("syn_collinear_dummy", list(dataset = "synth", rvar = "yb", evar = c("x1", "grp_f", "is_alpha"), lev = "yes"),
         note = "is_alpha equals the grp_f alpha dummy: R drops is_alpha")
log_case("syn_const", list(dataset = "synth", rvar = "yb", evar = c("x1", "const"), lev = "yes"),
         note = "a constant column: radiant refuses the model")
log_case("syn_quasi_sep", list(dataset = "synth", rvar = "yb", evar = c("x1", "qsep"), lev = "yes"),
         note = "quasi-complete separation: every row of level pure is a yes")
log_case("sep", list(dataset = "sep", rvar = "y", evar = c("x", "z"), lev = "yes"),
         note = "perfect separation: y is yes exactly when x > 0")

writeLines(capture.output(sessionInfo()), file.path(OUT, "est_sessionInfo.txt"))
cat("wrote", N_CASES, "estimation fixtures\n")
