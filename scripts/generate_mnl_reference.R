#!/usr/bin/env Rscript
# Generate reference fixtures for pyrsm.model.mnl (port of radiant.model::mnl).
#
# Writes coefficients, fit statistics, the printed summary (with confidence
# intervals, RRRs, the confusion matrix and a variables test), and predictions
# for a set of representative models to tests/reference/radiant_model/ as JSON.
#
# Usage:  Rscript scripts/generate_mnl_reference.R

suppressMessages({
  library(radiant.model)
  library(radiant.data)
  library(jsonlite)
})

OUT <- file.path("tests", "reference", "radiant_model")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

write_fixture <- function(name, obj) {
  path <- file.path(OUT, paste0(name, ".json"))
  writeLines(toJSON(obj, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), path)
  cat("wrote", path, "\n")
}

frame_list <- function(df) {
  df <- as.data.frame(df, stringsAsFactors = FALSE)
  lapply(df, function(x) if (is.factor(x)) as.character(x) else x)
}

dump_mnl <- function(name, args, sum_check = c("confint", "rrr", "confusion"),
                     test_var = "", pred_cmd = "") {
  res <- do.call(mnl, c(args, list(envir = globalenv())))
  if (is.character(res)) stop(paste(name, ":", res))
  m <- res$model
  ll <- as.numeric(res$model$logLik)
  null_dev <- as.numeric(res$model$null.deviance)
  edf <- m$edf
  nobs <- res$model$nobs
  ci <- confint(m, level = 0.95)
  if (length(dim(ci)) > 2) ci <- apply(ci, 2, rbind)
  out <- list(
    case = name,
    args = args,
    sum_check = sum_check,
    test_var = test_var,
    lev = res$lev,
    evar = res$evar,
    coeff = frame_list(res$coeff[, c("level", "label", "RRR", "coefficient", "std.error", "z.value", "p.value")]),
    ci_low = as.numeric(ci[, 1]),
    ci_high = as.numeric(ci[, 2]),
    loglik = ll,
    null_deviance = null_dev,
    deviance = m$deviance,
    edf = edf,
    AIC = m$AIC,
    nobs = nobs,
    convergence = m$convergence,
    fitted = frame_list(head(as.data.frame(fitted(m)), 25)),
    summary = paste(capture.output(summary(res, sum_check = sum_check, test_var = test_var)), collapse = "\n"),
    pred_data = frame_list(head(predict(res, pred_data = args$dataset, envir = globalenv()), 25))
  )
  if (!is.empty(pred_cmd)) {
    out$pred_cmd <- pred_cmd
    out$pred_cmd_result <- frame_list(predict(res, pred_cmd = pred_cmd, envir = globalenv()))
  }
  write_fixture(paste0("mnl_", name), out)
}

for (nm in c("ketchup", "titanic")) assign(nm, get(nm), envir = globalenv())

## titanic aggregated to frequency weights: same estimates as the full data
titanic_agg <- titanic %>%
  dplyr::count(pclass, sex, survived) %>%
  as.data.frame()
titanic_agg$n <- as.integer(titanic_agg$n)
write.csv(titanic_agg, file.path(OUT, "titanic_agg_mnl.csv"), row.names = FALSE)
assign("titanic_agg", titanic_agg, envir = globalenv())

prices <- c("price.heinz28", "price.heinz32", "price.heinz41", "price.hunts32")
disps <- c("disp.heinz28", "disp.heinz32", "disp.heinz41", "disp.hunts32")
feats <- c("feat.heinz28", "feat.heinz32", "feat.heinz41", "feat.hunts32")

dump_mnl("ketchup_price",
  list(dataset = "ketchup", rvar = "choice", evar = prices, lev = "heinz28"),
  pred_cmd = "price.heinz28 = seq(3.40, 5.20, 0.1)")
dump_mnl("ketchup_all",
  list(dataset = "ketchup", rvar = "choice", evar = c(prices, disps, feats), lev = "heinz28"),
  test_var = disps,
  pred_cmd = "disp.heinz41 = c('yes', 'no'), feat.heinz41 = c('yes', 'no')")
dump_mnl("ketchup_base_hunts32",
  list(dataset = "ketchup", rvar = "choice", evar = prices, lev = "hunts32"))
dump_mnl("ketchup_int",
  list(dataset = "ketchup", rvar = "choice", evar = c("price.heinz28", "price.heinz32", "disp.heinz28"),
       lev = "heinz28", int = "price.heinz28:disp.heinz28"),
  test_var = "price.heinz28")
dump_mnl("ketchup_std",
  list(dataset = "ketchup", rvar = "choice", evar = c(prices, "disp.heinz28"), lev = "heinz28",
       check = "standardize"),
  pred_cmd = "price.heinz28 = c(4, 5)")
dump_mnl("ketchup_center",
  list(dataset = "ketchup", rvar = "choice", evar = prices, lev = "heinz28", check = "center"))
dump_mnl("ketchup_noint",
  list(dataset = "ketchup", rvar = "choice", evar = prices, lev = "heinz28", check = "no_int"))
dump_mnl("ketchup_filter",
  list(dataset = "ketchup", rvar = "choice", evar = prices, lev = "heinz28",
       data_filter = "price.heinz28 > 4"))
dump_mnl("ketchup_stepwise",
  list(dataset = "ketchup", rvar = "choice", evar = c(prices, disps, feats), lev = "heinz28",
       check = "stepwise-backward"),
  sum_check = "")
dump_mnl("titanic_pclass",
  list(dataset = "titanic", rvar = "pclass", evar = c("sex", "age", "survived"), lev = "1st"))
dump_mnl("titanic_agg_wts",
  list(dataset = "titanic_agg", rvar = "pclass", evar = c("sex", "survived"), lev = "1st", wts = "n"))
dump_mnl("titanic_unagg",
  list(dataset = "titanic", rvar = "pclass", evar = c("sex", "survived"), lev = "1st"))

writeLines(capture.output(sessionInfo()), file.path(OUT, "mnl_sessionInfo.txt"))
