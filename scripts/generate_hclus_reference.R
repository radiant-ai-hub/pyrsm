#!/usr/bin/env Rscript
# Reference fixtures for every hclus distance and linkage method in Radiant.
#
# Radiant's UI offers the distances sq.euclidian, binary, canberra, euclidean,
# gower, manhattan, maximum, minkowski and the methods ward.D ("Ward's"),
# single, complete, average, mcquitty, median, centroid. Each distance is run
# with a few methods and sq.euclidian with every method (plus ward.D2, which
# hclust accepts although the UI does not list it). Merge heights (in R's merge
# order), the merge matrix, cluster memberships and the dissimilarities R's
# dist() produced are stored.
#
# Binary distances on (standardized) ratings are all zero, because dist()
# treats every non-zero value as "on". The binary cases therefore use
# shopping_bin = pmax(v - 4, 0), which mixes zeros with values 1..3: R counts
# any non-zero pair as agreeing, unlike SciPy's jaccard, which compares values.
#
# Usage:  Rscript scripts/generate_hclus_reference.R

suppressMessages({
  library(radiant.multivariate)
  library(jsonlite)
})

OUT <- file.path("tests", "reference", "radiant_multivariate")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

data(shopping)
shopping_bin <- mutate(shopping, across(v1:v6, ~ pmax(.x - 4, 0)))
distances <- c("sq.euclidian", "binary", "canberra", "euclidean", "gower",
               "manhattan", "maximum", "minkowski")
methods <- c("ward.D", "single", "complete", "average", "mcquitty", "median",
             "centroid", "ward.D2")

grid <- list()
for (d in distances) {
  ms <- if (d == "sq.euclidian") {
    methods
  } else if (d == "binary") {
    c("single", "complete")
  } else {
    c("ward.D", "complete", "average", "mcquitty")
  }
  for (m in ms) grid[[length(grid) + 1]] <- c(d, m)
}

cases <- list()
for (dm in grid) {
  for (std in c(TRUE, FALSE)) {
    if (dm[1] == "binary") {
      if (std) next
      res <- hclus(shopping_bin, "v1:v6", distance = dm[1], method = dm[2], standardize = std)
    } else {
      res <- hclus(shopping, "v1:v6", distance = dm[1], method = dm[2], standardize = std)
    }
    hc <- res$hc_out
    cases[[paste(dm[1], dm[2], std, sep = "_")]] <- list(
      distance = dm[1],
      method = dm[2],
      standardize = std,
      data = if (dm[1] == "binary") "shopping_bin" else "shopping",
      dist = as.numeric(res$d),
      height = as.numeric(hc$height),
      merge = as.integer(t(hc$merge)),
      cutree = lapply(setNames(2:4, 2:4), function(k) as.integer(cutree(hc, k)))
    )
  }
}

path <- file.path(OUT, "hclus_shopping_distances.json")
writeLines(toJSON(list(dataset = "shopping", vars = paste0("v", 1:6), cases = cases),
                  auto_unbox = TRUE, digits = NA, null = "null", na = "null"), path)
cat("wrote", path, "\n")

# ---- kclus: K-means initialized from hclus with each distance / method -----

# kclus(hc_init = TRUE) takes the same distance and method choices; fun =
# "kproto" without categorical variables switches to K-means but keeps the
# Gower distance it forces for the initial hclus.
kcases <- list()
kgrid <- list(
  c("kmeans", "sq.euclidian", "ward.D"), c("kmeans", "sq.euclidian", "mcquitty"),
  c("kmeans", "sq.euclidian", "centroid"), c("kmeans", "maximum", "complete"),
  c("kmeans", "canberra", "average"), c("kmeans", "minkowski", "median"),
  c("kmeans", "manhattan", "single"), c("kmeans", "gower", "ward.D"),
  c("kproto", "sq.euclidian", "ward.D")
)
for (g in kgrid) {
  for (k in 2:4) {
    res <- kclus(shopping, "v1:v6", fun = g[1], hc_init = TRUE, distance = g[2],
                 method = g[3], nr_clus = k)
    km <- res$km_out
    kcases[[paste(g[1], g[2], g[3], k, sep = "_")]] <- list(
      fun = g[1], distance = g[2], method = g[3], nr_clus = k,
      used_fun = res$fun, used_distance = res$distance,
      cluster = as.integer(km$cluster),
      sizes = as.integer(km$size),
      withinss = as.numeric(km$withinss),
      tot.withinss = as.numeric(km$tot.withinss),
      betweenss = as.numeric(km$betweenss)
    )
  }
}
path <- file.path(OUT, "kclus_shopping_hc_init.json")
writeLines(toJSON(list(dataset = "shopping", vars = paste0("v", 1:6), cases = kcases),
                  auto_unbox = TRUE, digits = NA, null = "null", na = "null"), path)
cat("wrote", path, "\n")

# ---- kclus: a single variable ------------------------------------------------

res <- kclus(shopping, "v1", nr_clus = 3)
km <- res$km_out
path <- file.path(OUT, "kclus_shopping_v1.json")
writeLines(toJSON(list(
  vars = "v1", nr_clus = 3,
  cluster = as.integer(km$cluster), sizes = as.integer(km$size),
  withinss = as.numeric(km$withinss), betweenss = as.numeric(km$betweenss),
  totss = as.numeric(km$totss), clus_means = as.numeric(res$clus_means$v1)
), auto_unbox = TRUE, digits = NA), path)
cat("wrote", path, "\n")
