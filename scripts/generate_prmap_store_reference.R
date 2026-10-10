#!/usr/bin/env Rscript
# Reference fixtures for storing prmap factor scores.
#
# prmap objects inherit from full_factor, so Radiant's "Store factor scores"
# runs store.full_factor: the *respondent-level* factor scores (fres$scores,
# one row per analysed row, not the brand means) are written to columns
# <name>1..<name>k, with NA for rows the filter or missing values excluded.
#
# Usage:  Rscript scripts/generate_prmap_store_reference.R

suppressMessages({
  library(radiant.multivariate)
  library(jsonlite)
})

OUT <- file.path("tests", "reference", "radiant_multivariate")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

data(computer)
data(retailers)
computer_na <- computer
computer_na$value[2] <- NA

dump <- function(dataset, brand, attr, pref = "", nr_dim = 2, data_filter = "", name = "pm") {
  res <- prmap(dataset, brand = brand, attr = attr, pref = pref, nr_dim = nr_dim,
               data_filter = data_filter)
  stored <- store(dataset, res, name = name)
  cols <- paste0(name, seq_len(nr_dim))
  list(
    brand = brand, attr = attr, pref = pref, nr_dim = nr_dim,
    data_filter = data_filter, name = name,
    columns = colnames(stored),
    stored = lapply(setNames(cols, cols), function(c) as.numeric(stored[[c]]))
  )
}

cases <- list(
  computer = dump(computer, "brand", "high_end:business"),
  computer_filter = dump(computer, "brand", "high_end:business", data_filter = "high_end > 4"),
  computer_na = dump(computer_na, "brand", "high_end:business"),
  computer_3 = dump(computer, "brand", "high_end:business", nr_dim = 3, name = "dim"),
  retailers_pref = dump(retailers, "retailer", "good_value:cluttered",
                        pref = "segment1:segment2", name = "f")
)

path <- file.path(OUT, "prmap_store.json")
writeLines(toJSON(cases, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), path)
cat("wrote", path, "\n")
