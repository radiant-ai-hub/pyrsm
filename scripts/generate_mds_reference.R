#!/usr/bin/env Rscript
# Reference fixtures for pyrsm.multivariate.mds from radiant.multivariate::mds.
#
# Every case runs radiant.multivariate::mds on the city, city2 and tpbrands
# example data (metric and non-metric, 2 and 3 dimensions, plus a filtered
# subset) and writes the coordinates, the stress, the original and recovered
# distances and the printed summary. Non-metric MDS is MASS::isoMDS started
# from cmdscale; its recovered distances are on the scale of the data, which is
# what tests/test_mds_radiant_parity.py checks.
#
# It also writes the internals pyrsm ports: isoMDS with a small iteration cap,
# and R's BFGS optimizer (optim method "BFGS" = vmmin, which isoMDS calls) on
# the Rosenbrock function.
#
# Usage:  Rscript scripts/generate_mds_reference.R

suppressMessages({
  library(radiant.multivariate)
  library(radiant.data)
  library(MASS)
  library(jsonlite)
})

OUT <- file.path("tests", "reference", "radiant_multivariate")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# flat, row-major; reshape in Python via (nrow, ncol)
mat_to_list <- function(m) {
  m <- as.matrix(m)
  storage.mode(m) <- "double"
  rn <- rownames(m)
  cn <- colnames(m)
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
  # I(17): 17 significant digits round-trip a double exactly (NA keeps 15)
  writeLines(toJSON(obj, auto_unbox = TRUE, digits = I(17), null = "null", na = "null"), path)
  cat("wrote", path, "\n")
}

data(city, city2, tpbrands, package = "radiant.multivariate")

dump_mds <- function(name, dataset, id1, id2, dis, method = "metric",
                     nr_dim = 2, seed = 1234, data_filter = "") {
  res <- mds(dataset, id1, id2, dis, method = method, nr_dim = nr_dim,
             seed = seed, data_filter = data_filter)
  write_fixture(paste0("mds_", name), list(
    case = name,
    method = method,
    nr_dim = nr_dim,
    data_filter = data_filter,
    nobs = res$nrObs,
    points = mat_to_list(res$res$points),
    stress = as.numeric(res$res$stress),
    orig_dist = mat_to_list(as.matrix(res$mds_dis_mat)),
    recovered_dist = mat_to_list(as.matrix(dist(res$res$points))),
    labels = rownames(res$res$points),
    summary = capture.output(summary(res, dec = 2))
  ))
}

DATA <- list(
  city = list(city, "from", "to", "distance"),
  city2 = list(city2, "from", "to", "distance"),
  tpbrands = list(tpbrands, "id1", "id2", "dissimilarity")
)
for (nm in names(DATA)) {
  a <- DATA[[nm]]
  for (method in c("metric", "non-metric")) {
    for (k in 2:3) {
      case <- paste0(nm, "_", sub("-", "", method), if (k == 3) "_3d" else "")
      dump_mds(case, a[[1]], a[[2]], a[[3]], a[[4]], method = method, nr_dim = k)
    }
  }
}

# a filter that reads the same in R and pyrsm: drop the zero self-distances,
# which leaves exactly the lower triangle (66 rows for 12 cities)
dump_mds("city2_nonmetric_filter", city2, "from", "to", "distance",
         method = "non-metric", data_filter = "distance > 0")

# ---- isoMDS internals ------------------------------------------------------

dis_matrix <- function(dataset, id1, id2, dis) {
  res <- mds(dataset, id1, id2, dis, method = "metric")
  res$mds_dis_mat
}

dump_isomds <- function(name, d, k = 2, maxit = 50, tol = 1e-6) {
  init <- cmdscale(d, k)
  res <- isoMDS(d, y = init, k = k, maxit = maxit, trace = FALSE, tol = tol)
  write_fixture(paste0("isomds_", name), list(
    case = name,
    k = k,
    maxit = maxit,
    tol = tol,
    dis = mat_to_list(as.matrix(d)),
    init = mat_to_list(init),
    points = mat_to_list(res$points),
    stress_pct = as.numeric(res$stress)
  ))
}

dump_isomds("tpbrands_maxit5", dis_matrix(tpbrands, "id1", "id2", "dissimilarity"), maxit = 5)
dump_isomds("tpbrands_maxit200", dis_matrix(tpbrands, "id1", "id2", "dissimilarity"), maxit = 200)
dump_isomds("city2_tol1e-3", dis_matrix(city2, "from", "to", "distance"), tol = 1e-3)
# radiant's own setting (maxit 50, tol 1e-6) on real distances: isoMDS stops
# near its stress floor (0.01%), where the last bits of the start decide when
city_d <- dis_matrix(city, "from", "to", "distance")
city2_d <- dis_matrix(city2, "from", "to", "distance")
dump_isomds("city_radiant", city_d, tol = 1e-6)
dump_isomds("city_3d_radiant", city_d, k = 3, tol = 1e-6)
dump_isomds("city2_radiant", city2_d, tol = 1e-6)
dump_isomds("city2_3d_radiant", city2_d, k = 3, tol = 1e-6)
dump_isomds("tpbrands_radiant", dis_matrix(tpbrands, "id1", "id2", "dissimilarity"), tol = 1e-6)
# a 1-7 rating scale where the starting map is already close: few iterations
dump_isomds("tpbrands_3d_maxit1", dis_matrix(tpbrands, "id1", "id2", "dissimilarity"), k = 3, maxit = 1)

# ---- R's BFGS (vmmin) on Rosenbrock ----------------------------------------

fr <- function(x) 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
grr <- function(x) c(-400 * x[1] * (x[2] - x[1]^2) - 2 * (1 - x[1]), 200 * (x[2] - x[1]^2))
bfgs <- function(maxit, reltol, abstol = -Inf) {
  o <- optim(c(-1.2, 1), fr, grr, method = "BFGS",
             control = list(maxit = maxit, reltol = reltol, abstol = abstol))
  list(maxit = maxit, reltol = reltol, abstol = if (is.finite(abstol)) abstol else NULL,
       par = o$par, value = o$value, fncount = unname(o$counts[1]),
       grcount = unname(o$counts[2]), convergence = o$convergence)
}
write_fixture("vmmin_rosenbrock", list(
  start = c(-1.2, 1),
  runs = list(bfgs(100, sqrt(.Machine$double.eps)), bfgs(10, 1e-8), bfgs(100, 1e-8, abstol = 1e-2))
))
