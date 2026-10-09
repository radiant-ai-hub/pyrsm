#!/usr/bin/env Rscript
# Reference fixtures for every full_factor rotation offered in Radiant's UI.
#
# radiant.multivariate::full_factor calls psych::principal (PCA) or psych::fa
# (fm = "ml") with the chosen rotation. psych names, orders and sign-flips the
# factors, so the fixtures keep the column names and order exactly as R prints
# them. For oblique rotations (promax, oblimin, simplimax) the factor
# correlations (Phi) are stored too.
#
# psych::fa (method "maxlik") tries 20 random starting rotations for the
# GPArotation criteria (quartimax, oblimin, simplimax) and keeps the "best"
# one, so for those cases R's result depends on the RNG state: the factor
# labels it prints can be permuted and, for simplimax, psych sometimes keeps a
# different local optimum (e.g. factor correlations near -0.86). For these
# cases the fixture records the modal outcome over seeds 1..40 (with its
# share); pyrsm starts from the identity rotation, which finds that solution.

suppressMessages({
  library(radiant.multivariate)
  library(jsonlite)
})

OUT <- file.path("tests", "reference", "radiant_multivariate")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

mat_to_list <- function(m) {
  m <- as.matrix(m)
  storage.mode(m) <- "double"
  list(
    values = as.numeric(t(m)),
    rownames = if (is.null(rownames(m))) NA else rownames(m),
    colnames = if (is.null(colnames(m))) NA else colnames(m),
    nrow = nrow(m),
    ncol = ncol(m)
  )
}

data(toothpaste)
rotations <- c("none", "varimax", "quartimax", "equamax", "promax", "oblimin", "simplimax")
cases <- list()
for (method in c("PCA", "maxlik")) {
  for (nf in 2:3) {
    for (rot in rotations) {
      seed <- 1
      share <- 1
      if (method == "maxlik" && rot %in% c("quartimax", "oblimin", "simplimax")) {
        outcome <- sapply(1:40, function(s) {
          set.seed(s)
          r <- full_factor(toothpaste, "v1:v6", method = method, nr_fact = nf, rotation = rot)
          paste(round(as.matrix(r$floadings), 3), collapse = ",")
        })
        modal <- names(which.max(table(outcome)))
        seed <- which(outcome == modal)[1]
        share <- mean(outcome == modal)
      }
      set.seed(seed)
      res <- full_factor(toothpaste, "v1:v6", method = method, nr_fact = nf, rotation = rot)
      key <- paste(method, nf, rot, sep = "_")
      cases[[key]] <- list(
        method = method,
        nr_fact = nf,
        rotation = rot,
        seed = seed,
        modal_share = share,
        floadings = mat_to_list(res$floadings),
        Phi = if (is.null(res$fres$Phi)) NULL else mat_to_list(res$fres$Phi),
        communality = as.numeric(1 - res$fres$uniquenesses),
        scores = mat_to_list(res$fres$scores),
        summary = paste(capture.output(summary(res)), collapse = "\n")
      )
    }
  }
}

path <- file.path(OUT, "full_factor_toothpaste_rotations.json")
writeLines(toJSON(list(dataset = "toothpaste", vars = paste0("v", 1:6), cases = cases),
                  auto_unbox = TRUE, digits = NA, null = "null", na = "null"), path)
cat("wrote", path, "\n")
