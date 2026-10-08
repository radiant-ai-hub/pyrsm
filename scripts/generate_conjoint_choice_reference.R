#!/usr/bin/env Rscript
# Reference fixtures for pyrsm.multivariate.conjoint_choice (conditional logit),
# computed with survival::clogit on the chocolate choice data.
#
# Usage:  Rscript scripts/generate_conjoint_choice_reference.R

suppressMessages({
  library(survival)
  library(jsonlite)
})

OUT <- file.path("tests", "reference", "radiant_multivariate")

## R has no parquet reader installed here, so export a csv with Python first
csv <- tempfile(fileext = ".csv")
system2("uv", c("run", "python", "-c", shQuote(paste0(
  "import polars as pl; pl.read_parquet('examples/data/multivariate/choc.parquet')",
  ".write_csv('", csv, "')"
))))
choc <- read.csv(csv)
choc$brand <- factor(choc$brand, levels = c("Hershey", "Lindt", "Godiva"))
choc$chocolate <- factor(choc$chocolate, levels = c("Milk", "Dark", "White"))
choc$nuts <- factor(choc$nuts, levels = c("No", "Yes"))

dump_case <- function(name, d, rhs) {
  f <- as.formula(paste("chosen ~", rhs, "+ strata(choice_id)"))
  m <- clogit(f, data = d, method = "exact")
  sm <- summary(m)$coefficients
  lp <- predict(m, type = "lp", reference = "sample")
  ## choice probabilities: softmax of the linear predictor within each task
  eu <- exp(lp - ave(lp, d$choice_id, FUN = max))
  prob <- eu / ave(eu, d$choice_id, FUN = sum)
  out <- list(
    case = name,
    nobs = nrow(d),
    ntasks = length(unique(d$choice_id)),
    names = rownames(sm),
    coef = unname(sm[, "coef"]),
    se = unname(sm[, "se(coef)"]),
    z = unname(sm[, "z"]),
    p = unname(sm[, "Pr(>|z|)"]),
    loglik = m$loglik,
    AIC = AIC(m),
    BIC = BIC(m),
    prob = unname(prob)
  )
  path <- file.path(OUT, paste0("conjoint_choice_", name, ".json"))
  writeLines(toJSON(out, auto_unbox = TRUE, digits = NA), path)
  cat("wrote", path, "\n")
}

dump_case("choc", choc, "brand + chocolate + price + nuts")
dump_case("choc_int", choc, "brand + chocolate + price + nuts + brand:price")

## pyrsm: data_filter = "resp <= 120", sort = "-choice_id", slice = "1:1500"
sub <- choc[choc$resp <= 120, ]
sub <- sub[order(-sub$choice_id), ][1:1500, ]
dump_case("choc_scope", sub, "brand + chocolate + price + nuts")
