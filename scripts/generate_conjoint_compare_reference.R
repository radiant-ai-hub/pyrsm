#!/usr/bin/env Rscript
# Generate reference fixtures for tests/test_conjoint_radiant_compare.py.
#
# Extends the conjoint fixtures written by generate_multivariate_reference.R
# (conjoint_{mp3,carpet,movie,mp3_int,mp3_by_radio}.json) with everything those
# do not cover: full regression statistics (t, p, adj. R-squared, F-test),
# multicollinearity diagnostics (car::vif GVIF), plot data (titles, x-axis level
# order, plot_ylim), radiant's own by= output (wide PW / IW tables), the data
# filter, character / numeric attributes, predict() with pred_data and/or
# pred_cmd (incl. by=), store() and the captured summary() text.
#
# Usage:  Rscript scripts/generate_conjoint_compare_reference.R

suppressMessages({
  library(radiant.multivariate)
  library(radiant.data)
  library(jsonlite)
})

OUT <- file.path("tests", "reference", "radiant_multivariate")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

write_fixture <- function(name, obj) {
  path <- file.path(OUT, paste0("conjoint_compare_", name, ".json"))
  writeLines(toJSON(obj, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), path)
  cat("wrote", path, "\n")
}

# data.frame -> list(columns = ..., data = list(col = values))
df_to_list <- function(df) {
  df <- as.data.frame(df, stringsAsFactors = FALSE)
  cols <- lapply(df, function(x) if (is.factor(x)) as.character(x) else as.vector(x))
  list(columns = colnames(df), data = cols)
}

quiet <- function(expr) suppressWarnings(expr)
out_txt <- function(expr) paste(quiet(capture.output(expr)), collapse = "\n")

coeff_list <- function(coeff) {
  list(
    label = as.character(coeff$label),
    coefficient = as.numeric(coeff$coefficient),
    std.error = as.numeric(coeff$std.error),
    t.value = as.numeric(coeff$t.value),
    p.value = as.numeric(coeff$p.value),
    sig_star = trimws(as.character(coeff$sig_star))
  )
}

fit_stats <- function(model) {
  g <- quiet(glance(model))
  list(
    rsq = g$r.squared,
    rsq_adj = g$adj.r.squared,
    fstat = as.numeric(g$statistic),
    f_pvalue = as.numeric(g$p.value),
    df_model = as.numeric(g$df), # broom >= 0.7: numerator df (excl. intercept)
    df_resid = as.numeric(g$df.residual),
    nobs = nrow(model$model)
  )
}

tab_list <- function(tab) {
  pw <- tab$PW
  list(
    PW = list(
      Attributes = as.character(pw$Attributes),
      Levels = as.character(pw$Levels),
      PW = as.numeric(pw$PW)
    ),
    IW = list(Attributes = as.character(tab$IW$Attributes), IW = as.numeric(tab$IW$IW)),
    plot_ylim = list(
      attr = rownames(tab$plot_ylim),
      Min = as.numeric(tab$plot_ylim$Min),
      Max = as.numeric(tab$plot_ylim$Max)
    )
  )
}

plot_info <- function(res, show = "") {
  pl <- quiet(plot(res, plots = c("pw", "iw"), show = show, custom = TRUE))
  lapply(pl, function(p) {
    bd <- quiet(ggplot_build(p))
    list(
      title = p$labels$title,
      x_labels = as.character(bd$layout$panel_params[[1]]$x$get_labels()),
      y = as.numeric(layer_data(p, 1)$y)
    )
  })
}

vif_list <- function(model) {
  v <- car::vif(model)
  if (!is.null(dim(v))) v <- v[, "GVIF"]
  list(names = names(v), VIF = as.numeric(v))
}

dump_model <- function(res, level = "full") {
  ml <- res$model_list[[level]]
  c(
    list(coeff = coeff_list(ml$coeff)),
    fit_stats(ml$model),
    tab_list(ml$tab)
  )
}

# ---- 1. full statistics for the three example datasets ---------------------

dump_full <- function(name, dataset, rvar, evar, int = "", reverse = FALSE,
                      data_filter = "") {
  res <- quiet(conjoint(dataset, rvar = rvar, evar = evar, int = int,
                        reverse = reverse, data_filter = data_filter))
  out <- c(
    list(case = name, rvar = rvar, evar = res$evar, int = int, reverse = reverse,
         data_filter = data_filter),
    dump_model(res),
    list(
      vif = vif_list(res$model_list$full$model),
      plots = plot_info(res),
      summary = out_txt(summary(res)),
      summary_additional = out_txt(summary(res, additional = TRUE, mc_diag = TRUE))
    )
  )
  write_fixture(name, out)
  invisible(res)
}

dump_full("full_mp3", mp3, "Rating", "Memory:Shape")
dump_full("full_carpet", carpet, "ranking", "design:money_back", reverse = TRUE)
dump_full("full_movie", movie, "Ranking", "price:food", reverse = TRUE)

# ---- 2. interactions -------------------------------------------------------

dump_full("int_mp3", mp3, "Rating", "Memory:Shape", int = "Memory:Shape")
dump_full("int_carpet_rev", carpet, "ranking", "design:money_back",
          int = "price:seal", reverse = TRUE)

# ---- 3. data filter (R conjoint has no arr / rows arguments) ---------------

dump_full("filter_mp3", mp3, "Rating", "Memory:Shape", data_filter = "Rating > 30")

# pyrsm-only sort / slice: reproduce the same rows in R by subsetting first
mp3_slice <- mp3[1:16, ]
dump_full("slice_mp3", mp3_slice, "Rating", "Memory:Shape")
mp3_sortslice <- dplyr::slice(dplyr::arrange(mp3, dplyr::desc(Rating)), 1:14)
dump_full("sortslice_mp3", mp3_sortslice, "Rating", "Memory:Shape")

# ---- 4. character / numeric attributes -------------------------------------

mp3_chr <- mp3
mp3_chr$Price <- as.character(mp3_chr$Price) # R: factor() -> alphabetical levels
res <- quiet(conjoint(mp3_chr, "Rating", "Memory:Shape"))
write_fixture("char_attr", c(list(case = "char_attr"), dump_model(res)))

mp3_num <- mp3
mp3_num$Price <- as.numeric(gsub("\\$", "", as.character(mp3_num$Price)))
res <- quiet(conjoint(mp3_num, "Rating", "Memory:Shape"))
ml <- res$model_list$full
write_fixture("num_attr", c(
  list(case = "num_attr", coeff = coeff_list(ml$coeff), PW_msg = ml$tab$PW),
  fit_stats(ml$model)
))

# ---- 5. by-group models ----------------------------------------------------

pred_cmd_full <- "Memory = c('4GB','8GB'); Price = c('$50','$150')"

dump_by <- function(name, dataset, rvar, evar, by, reverse = FALSE,
                    pred_cmd = "") {
  res <- quiet(conjoint(dataset, rvar = rvar, evar = evar, by = by, reverse = reverse))
  groups <- list()
  for (lev in res$bylevs) {
    groups[[lev]] <- c(
      dump_model(res, lev),
      list(
        plots = plot_info(res, show = lev),
        summary = out_txt(summary(res, show = lev)),
        summary_additional = out_txt(summary(res, show = lev, additional = TRUE))
      )
    )
  }
  out <- list(
    case = name, rvar = rvar, evar = res$evar, by = by, reverse = reverse,
    bylevs = res$bylevs,
    PW_wide = df_to_list(res$PW),
    IW_wide = df_to_list(res$IW),
    groups = groups
  )
  pd <- quiet(predict(res, pred_data = dataset))
  if (!is.character(pd)) out$pred_data <- df_to_list(pd)
  if (!is.empty(pred_cmd)) {
    pc <- quiet(predict(res, pred_cmd = pred_cmd))
    out$pred_cmd <- pred_cmd
    out$pred_cmd_result <- df_to_list(pc)
  }
  write_fixture(name, out)
}

dump_by("by_mp3_radio", mp3, "Rating", c("Memory", "Price"), "Radio",
        pred_cmd = pred_cmd_full)
dump_by("by_mp3_radio_rev", mp3, "Rating", c("Memory", "Price"), "Radio",
        reverse = TRUE)
dump_by("by_movie_sight_rev", movie, "Ranking", c("price", "comfort", "food"), "sight",
        reverse = TRUE)

# by variable inside the evar range: radiant drops it from evar
res <- quiet(conjoint(mp3, "Rating", "Memory:Shape", by = "Radio"))
write_fixture("by_in_evar", list(
  case = "by_in_evar", evar = res$evar, bylevs = res$bylevs,
  PW_wide = df_to_list(res$PW), IW_wide = df_to_list(res$IW),
  coeff_No = coeff_list(res$model_list$No$coeff),
  coeff_Yes = coeff_list(res$model_list$Yes$coeff)
))

# non-factor by variable: radiant's as_factor() == factor() -> sorted levels
mp3_grp <- mp3
mp3_grp$grp <- ifelse(mp3_grp$Radio == "No", "z", "a")
res <- quiet(conjoint(mp3_grp, "Rating", c("Memory", "Price"), by = "grp"))
write_fixture("by_char", list(
  case = "by_char", bylevs = res$bylevs, PW_wide = df_to_list(res$PW)
))

# ---- 6. predict / store (no by) --------------------------------------------

res <- quiet(conjoint(mp3, "Rating", "Memory:Shape"))
res_int <- quiet(conjoint(mp3, "Rating", "Memory:Shape", int = "Memory:Shape"))
cmd_all <- "Memory = c('4GB','8GB'); Radio = 'Yes'; Size = 'Small'; Price = '$50'; Shape = 'Square'"
cmd_partial <- "Memory = c('4GB','8GB')"
cmd_comma <- "Memory = c('4GB','8GB'), Price = '$50'"
cmd_mutate <- "Memory = '8GB'"

p_data <- predict(res, pred_data = mp3)
p_conf <- predict(res, pred_data = mp3, se = TRUE)
p_pred90 <- predict(res, pred_data = mp3, se = TRUE, interval = "prediction", conf_lev = 0.9)
p_cmd_all <- predict(res, pred_cmd = cmd_all)
p_cmd_partial <- predict(res, pred_cmd = cmd_partial)
p_cmd_comma <- predict(res, pred_cmd = cmd_comma)
p_data_cmd <- predict(res, pred_data = mp3, pred_cmd = cmd_mutate)
p_int <- predict(res_int, pred_data = mp3, se = TRUE)

st_single <- store(mp3, p_conf, name = "pref")
st_triple <- store(mp3, p_conf, name = "pref, lb, ub")

write_fixture("predict", list(
  cmd_all = cmd_all, cmd_partial = cmd_partial, cmd_comma = cmd_comma,
  cmd_mutate = cmd_mutate,
  pred_data = df_to_list(p_data),
  pred_conf = df_to_list(p_conf),
  pred_pred90 = df_to_list(p_pred90),
  pred_cmd_all = df_to_list(p_cmd_all),
  pred_cmd_partial = df_to_list(p_cmd_partial),
  pred_cmd_comma = df_to_list(p_cmd_comma),
  pred_data_cmd = df_to_list(p_data_cmd),
  pred_int = df_to_list(p_int),
  print_conf = out_txt(print(p_conf, n = 3)),
  store_single_cols = colnames(st_single),
  store_single = as.numeric(st_single$pref),
  store_triple_cols = colnames(st_triple),
  store_triple_lb = as.numeric(st_triple$lb),
  store_triple_ub = as.numeric(st_triple$ub)
))

cat("done\n")
