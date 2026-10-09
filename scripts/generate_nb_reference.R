#!/usr/bin/env Rscript
# Generate reference fixtures for pyrsm.model.nb (port of radiant.model::nb,
# which estimates with e1071::naiveBayes).
#
# Writes the a-priori counts, conditional probability tables (or means and
# standard deviations), predictions, variable-importance plot data and the
# printed summary for a set of representative models to
# tests/reference/radiant_model/nb_*.json.
#
# Usage:  Rscript scripts/generate_nb_reference.R

suppressMessages({
  library(radiant.model)
  library(radiant.data)
  library(jsonlite)
})

OUT <- file.path("tests", "reference", "radiant_model")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

mat_to_list <- function(m) {
  rn <- rownames(m)
  cn <- colnames(m)
  m <- unclass(as.matrix(m))
  storage.mode(m) <- "double"
  list(
    values = as.numeric(t(m)),
    rownames = if (is.null(rn)) NA else as.character(rn),
    colnames = if (is.null(cn)) NA else as.character(cn),
    nrow = nrow(m),
    ncol = ncol(m)
  )
}

write_fixture <- function(name, obj) {
  path <- file.path(OUT, paste0(name, ".json"))
  writeLines(toJSON(obj, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), path)
  cat("wrote", path, "\n")
}

## predictions as a list of columns (explanatory variables as character or
## numbers, one probability column per response level)
pred_to_list <- function(pred, evar, lev) {
  df <- as.data.frame(pred, stringsAsFactors = FALSE)
  out <- list()
  for (v in evar) {
    x <- df[[v]]
    out[[v]] <- if (is.numeric(x)) as.numeric(x) else as.character(x)
  }
  for (l in lev) out[[l]] <- as.numeric(df[[l]])
  out
}

dump_nb <- function(name, args, pred_cmd = "", vimp_lev = NULL) {
  ## pass the dataset by name so the summary shows "Data : <name>"
  res <- do.call(nb, c(args, list(envir = globalenv())))
  if (is.character(res)) stop(paste(name, ":", res))
  m <- res$model
  tables <- lapply(names(m$tables), function(v) {
    t <- m$tables[[v]]
    c(mat_to_list(t), list(var = v, numeric = unname(m$isnumeric[v])))
  })
  pred <- predict(res, pred_data = get(args$dataset, envir = globalenv()))
  vimp <- plot(res, plots = "vimp")$data
  out <- list(
    case = name,
    args = args,
    nobs = nrow(m$model),
    lev = res$lev,
    evar = res$evar,
    apriori = list(values = as.numeric(m$apriori), names = names(m$apriori)),
    tables = tables,
    summary = capture.output(summary(res)),
    prediction = pred_to_list(pred, res$evar, res$lev),
    vimp = lapply(as.list(vimp), function(x) if (is.factor(x)) as.character(x) else x)
  )
  if (!is.null(vimp_lev)) {
    out$vimp_lev <- vimp_lev
    vl <- plot(res, plots = "vimp", lev = vimp_lev)$data
    out$vimp_lev_data <- lapply(as.list(vl), function(x) if (is.factor(x)) as.character(x) else x)
  }
  if (!all(pred_cmd == "")) {
    pc <- predict(res, pred_cmd = pred_cmd)
    if (is.character(pc)) stop(paste(name, ":", pc))
    out$pred_cmd <- pred_cmd
    out$pred_cmd_prediction <- pred_to_list(pc, res$evar, res$lev)
  }
  write_fixture(paste0("nb_", name), out)
}

## copy the lazy-loaded package datasets into the global environment so
## nb can find them by name
for (nm in c("titanic", "diamonds", "dvd", "ketchup")) assign(nm, get(nm), envir = globalenv())

## a dataset with long factor labels (so printed tables wrap at 80 characters)
## and a numeric variable that is constant within one class (so e1071's
## threshold for a zero standard deviation is used)
titanic_wide <- titanic[, c("survived", "pclass", "sex", "age", "fare")]
titanic_wide$fare_bracket <- cut(
  titanic$fare,
  breaks = c(-1, 7.75, 8.05, 13, 26, 52, 600),
  labels = paste0("fare_bracket_", c("lowest", "low", "middle_low", "middle_high", "high", "highest"))
)
titanic_wide$age_yes <- ifelse(titanic$survived == "Yes", 30, titanic$age)
write.csv(titanic_wide, file.path(OUT, "nb_titanic_wide.csv"), row.names = FALSE)
assign("titanic_wide", titanic_wide, envir = globalenv())

dump_nb("titanic",
  list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex", "age")),
  pred_cmd = "pclass = levels(pclass)", vimp_lev = "No")
dump_nb("titanic_laplace",
  list(dataset = "titanic", rvar = "survived",
       evar = c("pclass", "sex", "age", "sibsp", "parch", "embarked"), laplace = 1),
  pred_cmd = "sex = levels(sex)")
dump_nb("titanic_cat",
  list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex", "embarked"), laplace = 0.5),
  pred_cmd = c("pclass = levels(pclass)", "sex = levels(sex)"))
dump_nb("titanic_pclass",
  list(dataset = "titanic", rvar = "pclass", evar = c("survived", "sex", "age", "fare")),
  pred_cmd = "age = seq(0, 80, 20)", vimp_lev = "2nd")
dump_nb("diamonds_cut",
  list(dataset = "diamonds", rvar = "cut", evar = c("price", "carat", "clarity")),
  pred_cmd = "clarity = levels(clarity)", vimp_lev = "Ideal")
dump_nb("ketchup",
  list(dataset = "ketchup", rvar = "choice",
       evar = c("price.heinz28", "disp.heinz28", "feat.heinz28", "price.hunts32")),
  pred_cmd = "price.heinz28 = seq(0.8, 1.6, 0.2)")
dump_nb("dvd",
  list(dataset = "dvd", rvar = "buy", evar = c("coupon", "purch", "last")),
  pred_cmd = "coupon = 0:1")
dump_nb("titanic_wide",
  list(dataset = "titanic_wide", rvar = "survived", evar = c("pclass", "fare_bracket", "age_yes")),
  pred_cmd = "fare_bracket = levels(fare_bracket)")

## data scope: filter (that removes a level of an explanatory variable),
## arrange and rows. Make sure there is no tie at the slice boundary.
chk <- get_data("diamonds", filt = "cut != 'Fair'", arr = "desc(price)", envir = globalenv())
stopifnot(chk$price[500] != chk$price[501])
dump_nb("diamonds_scope",
  list(dataset = "diamonds", rvar = "clarity", evar = c("cut", "color", "carat"),
       data_filter = "cut != 'Fair'", arr = "desc(price)", rows = 1:500),
  pred_cmd = "cut = levels(cut)")
chk <- get_data("titanic", filt = "age > 10", arr = "desc(fare)", envir = globalenv())
stopifnot(chk$fare[585] != chk$fare[586])
dump_nb("titanic_scope",
  list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex", "age"),
       data_filter = "age > 10", arr = "desc(fare)", rows = 1:585, laplace = 1))

writeLines(capture.output(sessionInfo()), file.path(OUT, "nb_sessionInfo.txt"))
