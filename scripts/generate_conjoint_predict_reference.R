#!/usr/bin/env Rscript
# Reference fixture for the columns and values conjoint predict() returns.
#
# R's predict.conjoint returns the profile (attribute) columns followed by
# Prediction (and interval columns with se = TRUE); with a `by` variable there
# is one block per by-level with the by column first. Covers Radiant's three
# prediction inputs: Data, Command and Data & Command.
#
# Usage:  Rscript scripts/generate_conjoint_predict_reference.R

suppressMessages({
  library(radiant.multivariate)
  library(jsonlite)
})

OUT <- file.path("tests", "reference", "radiant_multivariate")

df_to_list <- function(df) {
  df <- as.data.frame(df, stringsAsFactors = FALSE)
  list(columns = colnames(df),
       data = lapply(df, function(x) if (is.factor(x)) as.character(x) else as.vector(x)))
}

data(mp3)
cmd <- "Memory = c('4GB','8GB'); Radio = c('Yes','No'); Size = 'Medium'; Price = '$100'; Shape = 'Square'"
cmd_partial <- "Price = c('$50','$150'); Memory = '8GB'"
cmd_by <- "Memory = c('4GB','8GB'); Price = '$50'"
# Radiant splits the command on ";" before calling predict
split_cmd <- function(x) strsplit(x, ";\\s*")[[1]]

res <- conjoint(mp3, "Rating", "Memory:Shape")
res_by <- conjoint(mp3, "Rating", c("Memory", "Price"), by = "Radio")
mp3_head <- mp3[1:4, ]

out <- list(
  cmd = cmd, cmd_partial = cmd_partial, cmd_by = cmd_by, data_cmd = "Memory = '8GB'",
  pred_cmd = df_to_list(predict(res, pred_cmd = split_cmd(cmd))),
  pred_cmd_partial = df_to_list(predict(res, pred_cmd = split_cmd(cmd_partial))),
  pred_cmd_se = df_to_list(predict(res, pred_cmd = split_cmd(cmd), se = TRUE)),
  pred_data = df_to_list(predict(res, pred_data = mp3_head)),
  pred_data_cmd = df_to_list(predict(res, pred_data = mp3_head, pred_cmd = "Memory = '8GB'")),
  pred_by_cmd = df_to_list(predict(res_by, pred_cmd = split_cmd(cmd_by))),
  pred_by_data = df_to_list(predict(res_by, pred_data = mp3_head))
)
path <- file.path(OUT, "conjoint_predict_columns.json")
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), path)
cat("wrote", path, "\n")
