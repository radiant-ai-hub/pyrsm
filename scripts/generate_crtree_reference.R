#!/usr/bin/env Rscript
# Generate reference fixtures for pyrsm.model.crtree (port of radiant.model::crtree).
#
# Writes the fitted rpart frame, splits, cp table, variable importance, printed
# summary, mermaid tree source and predictions for a set of representative
# models to tests/reference/radiant_model/ as JSON.
#
# Usage:  Rscript scripts/generate_crtree_reference.R

suppressMessages({
  library(radiant.model)
  library(radiant.data)
  library(jsonlite)
})

OUT <- file.path("tests", "reference", "radiant_model")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

mat_to_list <- function(m) {
  m <- as.matrix(m)
  storage.mode(m) <- "double"
  list(
    values = as.numeric(t(m)),
    rownames = if (is.null(rownames(m))) NA else as.character(rownames(m)),
    colnames = if (is.null(colnames(m))) NA else as.character(colnames(m)),
    nrow = nrow(m),
    ncol = ncol(m)
  )
}

write_fixture <- function(name, obj) {
  path <- file.path(OUT, paste0(name, ".json"))
  writeLines(toJSON(obj, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), path)
  cat("wrote", path, "\n")
}

dump_crtree <- function(name, args, pred_cmd = "") {
  ## pass the dataset by name so the summary shows "Data : <name>"
  res <- do.call(crtree, c(args, list(envir = globalenv())))
  if (is.character(res)) stop(paste(name, ":", res))
  m <- res$model
  fr <- m$frame
  frame <- list(
    id = as.integer(rownames(fr)),
    var = as.character(fr$var),
    n = fr$n,
    wt = fr$wt,
    dev = fr$dev,
    yval = fr$yval,
    complexity = fr$complexity,
    ncompete = fr$ncompete,
    nsurrogate = fr$nsurrogate
  )
  if (!is.null(fr$yval2)) frame$yval2 <- mat_to_list(fr$yval2)
  pred <- predict(res, pred_data = get(args$dataset, envir = globalenv()))
  diagram <- plot(res)$x$diagram
  out <- list(
    case = name,
    args = args,
    nobs = nrow(m$model),
    lev = res$lev,
    type = res$type,
    frame = frame,
    splits = if (is.null(m$splits)) NA else mat_to_list(m$splits),
    csplit = if (is.null(m$csplit)) NA else mat_to_list(m$csplit),
    cptable = mat_to_list(m$cptable),
    importance = if (is.null(m$variable.importance)) NA else
      list(values = unname(m$variable.importance), names = names(m$variable.importance)),
    where = unname(m$where),
    summary = capture.output(summary(res)),
    summary_extra = capture.output(summary(res, splits = TRUE, cptab = TRUE)),
    diagram = diagram,
    prediction = pred$Prediction
  )
  if (pred_cmd != "") {
    pc <- predict(res, pred_cmd = pred_cmd)
    out$pred_cmd <- pred_cmd
    out$pred_cmd_prediction <- pc$Prediction
  }
  write_fixture(paste0("crtree_", name), out)
}

## copy the lazy-loaded package datasets into the global environment so
## crtree can find them by name
for (nm in c("titanic", "diamonds", "dvd", "ketchup")) assign(nm, get(nm), envir = globalenv())

titanic_agg <- titanic %>%
  dplyr::count(pclass, sex, survived, name = "n") %>%
  as.data.frame()
titanic_agg$n <- as.integer(titanic_agg$n)
write.csv(titanic_agg, file.path(OUT, "titanic_agg.csv"), row.names = FALSE)

dump_crtree("titanic_pclass_sex",
  list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex"), lev = "Yes"),
  pred_cmd = "pclass = levels(pclass)")
dump_crtree("titanic_age",
  list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex", "age"), lev = "Yes"))
dump_crtree("titanic_all",
  list(dataset = "titanic", rvar = "survived",
       evar = c("pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"),
       lev = "Yes", cp = 0.005))
dump_crtree("titanic_prior",
  list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex", "age"), lev = "Yes", prior = 0.5))
dump_crtree("titanic_cost",
  list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex", "age"), lev = "Yes",
       cost = 1, margin = 5))
dump_crtree("titanic_info",
  list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex", "age"), lev = "Yes",
       split = "information"))
dump_crtree("titanic_nodes",
  list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex", "age"), lev = "Yes", nodes = 4))
dump_crtree("titanic_pcp",
  list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex", "age"), lev = "No", pcp = 0.01))
dump_crtree("titanic_wts",
  list(dataset = "titanic_agg", rvar = "survived", evar = c("pclass", "sex"), lev = "Yes", wts = "n"))
dump_crtree("dvd",
  list(dataset = "dvd", rvar = "buy", evar = c("coupon", "purch", "last"), cp = 0.01))
dump_crtree("diamonds_reg",
  list(dataset = "diamonds", rvar = "price", evar = c("carat", "clarity", "cut"), type = "regression"))
dump_crtree("diamonds_reg_color",
  list(dataset = "diamonds", rvar = "price", evar = c("carat", "color"), type = "regression", cp = 0.005,
       minsplit = 20))
dump_crtree("diamonds_cut_multi",
  list(dataset = "diamonds", rvar = "cut", evar = c("clarity", "color", "carat"), cp = 0.01))
dump_crtree("ketchup_multi",
  list(dataset = "ketchup", rvar = "choice",
       evar = c("price.heinz28", "disp.heinz28", "feat.heinz28", "price.hunts32"), cp = 0.005))
dump_crtree("diamonds_scope",
  list(dataset = "diamonds", rvar = "price", evar = c("carat", "clarity"), type = "regression",
       data_filter = "cut != 'Fair'", rows = 1:1500, cp = 0.005))
dump_crtree("titanic_scope_arr",
  list(dataset = "titanic", rvar = "survived", evar = c("pclass", "sex", "age"), lev = "Yes",
       data_filter = "age > 10", arr = "desc(fare)", rows = 1:573))

# R's RNG, used for the cross-validation groups
set.seed(1234)
xg <- sample(rep(1:10, length.out = 1043), 1043, replace = FALSE)
set.seed(1234)
cvg <- sample(5, 1000, replace = TRUE)
write_fixture("r_rng", list(xgroups_1043_10 = xg, sample_5_1000 = cvg))

writeLines(capture.output(sessionInfo()), file.path(OUT, "sessionInfo.txt"))
