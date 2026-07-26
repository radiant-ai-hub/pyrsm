#!/usr/bin/env Rscript
# Generate reference fixtures for the pyrsm.multivariate port.
#
# Runs representative examples for every radiant.multivariate menu entry and
# writes numeric outputs to tests/reference/radiant_multivariate/ as JSON so the
# Python test-suite can validate exact parity.
#
# Usage:  Rscript scripts/generate_multivariate_reference.R

suppressMessages({
  library(radiant.multivariate)
  library(radiant.data)
  library(psych)
  library(jsonlite)
  library(polycor)
  library(clustMixType)
})

set.seed(1234)

OUT <- file.path("tests", "reference", "radiant_multivariate")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# ---- serialization helpers -------------------------------------------------

# Serialize a matrix / data.frame keeping row and column names.
mat_to_list <- function(m) {
  m <- as.matrix(m)
  storage.mode(m) <- "double"
  rn <- rownames(m)
  cn <- colnames(m)
  list(
    # flat, row-major; reshape in Python via (nrow, ncol)
    values = as.numeric(t(m)),
    rownames = if (is.null(rn)) NA else as.character(rn),
    colnames = if (is.null(cn)) NA else as.character(cn),
    nrow = nrow(m),
    ncol = ncol(m)
  )
}

vec_to_list <- function(v) {
  list(values = as.numeric(unname(v)), names = if (is.null(names(v))) NA else names(v))
}

write_fixture <- function(name, obj) {
  path <- file.path(OUT, paste0(name, ".json"))
  writeLines(toJSON(obj, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), path)
  cat("wrote", path, "\n")
}

# ---- pre_factor ------------------------------------------------------------

dump_pre_factor <- function(name, dataset, vars, hcor = FALSE) {
  res <- pre_factor(dataset, vars, hcor = hcor)
  out <- list(
    case = name,
    vars = res$vars,
    nobs = nrow(res$dataset),
    hcor = hcor,
    cmat = mat_to_list(res$cmat),
    bartlett = list(
      chisq = as.numeric(res$btest$chisq),
      df = as.numeric(res$btest$df),
      p.value = as.numeric(res$btest$p.value)
    ),
    kmo = as.numeric(res$pre_kmo$MSA),
    msai = vec_to_list(res$pre_kmo$MSAi),
    eigen = as.numeric(res$pre_eigen),
    pre_r2 = mat_to_list(res$pre_r2)
  )
  write_fixture(paste0("pre_factor_", name), out)
}

data(shopping); data(toothpaste); data(computer); data(retailers)
data(mp3); data(carpet); data(movie); data(city); data(tpbrands)
data(diamonds, package = "radiant.data")

dump_pre_factor("shopping", shopping, "v1:v6")
dump_pre_factor("toothpaste", toothpaste, "v1:v6")
dump_pre_factor("diamonds", diamonds, c("price", "carat", "table"))

# ordered-categorical polychoric path (validates polychoric reuse).
# radiant's pre_factor(hcor=TRUE) relies on polycor::hetcor which fails on this
# data and silently falls back to Pearson, so we build the reference directly
# from pairwise polycor::polychor (the 2-step estimator pyrsm reproduces) and
# run the standard pre_factor math on that matrix.
poly_matrix <- function(df) {
  k <- ncol(df)
  m <- diag(k)
  for (i in 1:(k - 1)) {
    for (j in (i + 1):k) {
      r <- polycor::polychor(df[[i]], df[[j]])
      m[i, j] <- m[j, i] <- r
    }
  }
  dimnames(m) <- list(colnames(df), colnames(df))
  m
}

dump_pre_factor_poly <- function(name, dataset, vars) {
  d <- get_data(dataset, vars)
  d <- as.data.frame(lapply(d, function(x) ordered(x)))
  cmat <- poly_matrix(d)
  btest <- psych::cortest.bartlett(cmat, nrow(d))
  pre_kmo <- psych::KMO(cmat)
  pre_eigen <- eigen(cmat)$values
  pre_r2 <- (1 - (1 / diag(solve(cmat))))
  out <- list(
    case = name,
    vars = colnames(d),
    nobs = nrow(d),
    hcor = TRUE,
    cmat = mat_to_list(cmat),
    bartlett = list(
      chisq = as.numeric(btest$chisq),
      df = as.numeric(btest$df),
      p.value = as.numeric(btest$p.value)
    ),
    kmo = as.numeric(pre_kmo$MSA),
    msai = vec_to_list(pre_kmo$MSAi),
    eigen = as.numeric(pre_eigen),
    pre_r2 = vec_to_list(stats::setNames(pre_r2, colnames(d)))
  )
  write_fixture(paste0("pre_factor_", name), out)
}

dump_pre_factor_poly("toothpaste_hcor", toothpaste, "v1:v6")

# ---- full_factor -----------------------------------------------------------

dump_full_factor <- function(name, dataset, vars, nr_fact = 1, rotation = "varimax",
                             method = "PCA", hcor = FALSE) {
  res <- full_factor(dataset, vars, method = method, hcor = hcor,
                     nr_fact = nr_fact, rotation = rotation)
  out <- list(
    case = name,
    vars = res$vars,
    nr_fact = nr_fact,
    rotation = rotation,
    method = method,
    hcor = hcor,
    floadings = mat_to_list(res$floadings),
    communality = vec_to_list(res$fres$communality),
    uniqueness = vec_to_list(res$fres$uniquenesses),
    eigen = as.numeric(res$fres$values),
    scores = mat_to_list(res$fres$scores)
  )
  write_fixture(paste0("full_factor_", name), out)
}

dump_full_factor("shopping_2", shopping, "v1:v6", nr_fact = 2, rotation = "varimax")
dump_full_factor("shopping_2_none", shopping, "v1:v6", nr_fact = 2, rotation = "none")
dump_full_factor("shopping_3", shopping, "v1:v6", nr_fact = 3, rotation = "varimax")
dump_full_factor("toothpaste_2", toothpaste, "v1:v6", nr_fact = 2, rotation = "varimax")
dump_full_factor("diamonds_1", diamonds, c("price", "carat", "table"), nr_fact = 1)

# PCA loadings from a polychoric correlation matrix (ordered-categorical + hcor).
# Scores in radiant use IRT for the all-categorical case and are out of scope;
# loadings/communalities/eigenvalues come straight from principal() on the
# polychoric matrix, which pyrsm reproduces.
dump_full_factor_poly <- function(name, dataset, vars, nr_fact = 2, rotation = "varimax") {
  d <- get_data(dataset, vars)
  d <- as.data.frame(lapply(d, function(x) ordered(x)))
  cmat <- poly_matrix(d)
  fres <- psych::principal(cmat, nfactors = nr_fact, rotate = rotation,
                           scores = FALSE, oblique.scores = FALSE)
  load <- unclass(fres$loadings)
  out <- list(
    case = name,
    vars = colnames(d),
    nr_fact = nr_fact,
    rotation = rotation,
    hcor = TRUE,
    floadings = mat_to_list(load),
    communality = vec_to_list(fres$communality),
    eigen = as.numeric(fres$values)
  )
  write_fixture(paste0("full_factor_", name), out)
}

dump_full_factor_poly("toothpaste_hcor", toothpaste, "v1:v6", nr_fact = 2)

# ---- hclus -----------------------------------------------------------------

dump_hclus <- function(name, dataset, vars, labels = "none",
                       distance = "sq.euclidian", method = "ward.D",
                       standardize = TRUE, cuts = c(2, 3)) {
  res <- hclus(dataset, vars, labels = labels, distance = distance,
               method = method, standardize = standardize)
  hc <- res$hc_out
  cut_list <- list()
  for (k in cuts) cut_list[[as.character(k)]] <- as.integer(cutree(hc, k))
  out <- list(
    case = name,
    vars = res$vars,
    labels = labels,
    distance = distance,
    method = method,
    standardize = standardize,
    height = as.numeric(hc$height),
    merge = mat_to_list(hc$merge),
    order = as.integer(hc$order),
    cutree = cut_list,
    labels_vec = if (!is.null(hc$labels)) as.character(hc$labels) else NA
  )
  write_fixture(paste0("hclus_", name), out)
}

dump_hclus("shopping", shopping, "v1:v6")
dump_hclus("toothpaste", toothpaste, "v1:v6")
dump_hclus("toothpaste_id", toothpaste, "v1:v6", labels = "id")

# ---- kclus -----------------------------------------------------------------

dump_kclus <- function(name, dataset, vars, nr_clus = 2, seed = 1234,
                       fun = "kmeans", hc_init = TRUE, standardize = TRUE) {
  res <- kclus(dataset, vars, fun = fun, hc_init = hc_init, seed = seed,
               nr_clus = nr_clus, standardize = standardize)
  km <- res$km_out
  out <- list(
    case = name,
    vars = res$vars,
    nr_clus = nr_clus,
    fun = fun,
    clus_means = mat_to_list(res$clus_means),
    sizes = as.integer(km$size),
    withinss = as.numeric(km$withinss),
    tot.withinss = as.numeric(km$tot.withinss),
    betweenss = as.numeric(km$betweenss),
    totss = as.numeric(km$totss),
    cluster = as.integer(km$cluster)
  )
  write_fixture(paste0("kclus_", name), out)
}

dump_kclus("shopping_2", shopping, "v1:v6", nr_clus = 2)
dump_kclus("shopping_3", shopping, "v1:v6", nr_clus = 3)
dump_kclus("toothpaste_3", toothpaste, "v1:v6", nr_clus = 3)

# ---- mds -------------------------------------------------------------------

dump_mds <- function(name, dataset, id1, id2, dis, method = "metric",
                     nr_dim = 2, seed = 1234) {
  res <- mds(dataset, id1, id2, dis, method = method, nr_dim = nr_dim, seed = seed)
  recovered <- as.matrix(dist(res$res$points))
  orig <- as.matrix(res$mds_dis_mat)
  out <- list(
    case = name,
    method = method,
    nr_dim = nr_dim,
    points = mat_to_list(res$res$points),
    stress = as.numeric(res$res$stress),
    orig_dist = mat_to_list(orig),
    recovered_dist = mat_to_list(recovered),
    labels = rownames(res$res$points)
  )
  write_fixture(paste0("mds_", name), out)
}

dump_mds("city_metric", city, "from", "to", "distance", method = "metric")
dump_mds("city_nonmetric", city, "from", "to", "distance", method = "non-metric")
dump_mds("tpbrands_metric", tpbrands, "id1", "id2", "dissimilarity", method = "metric")
dump_mds("tpbrands_nonmetric", tpbrands, "id1", "id2", "dissimilarity", method = "non-metric")

# ---- prmap -----------------------------------------------------------------

dump_prmap <- function(name, dataset, brand, attr, pref = "", nr_dim = 2, hcor = FALSE) {
  res <- prmap(dataset, brand = brand, attr = attr, pref = pref,
               nr_dim = nr_dim, hcor = hcor)
  out <- list(
    case = name,
    brand = brand,
    nr_dim = nr_dim,
    hcor = hcor,
    scores = mat_to_list(res$scores),
    loadings = mat_to_list(unclass(res$fres$loadings)),
    communality = vec_to_list(1 - res$fres$uniqueness),
    eigen = as.numeric(res$fres$values)
  )
  if (!is.null(res$pref_cor) && !identical(pref, "")) {
    out$pref_cor <- mat_to_list(res$pref_cor)
  }
  write_fixture(paste0("prmap_", name), out)
}

dump_prmap("computer", computer, "brand", "high_end:business")
dump_prmap("retailers", retailers, "retailer",
           "good_value:cluttered", pref = "segment1:segment2")

# ---- conjoint --------------------------------------------------------------

dump_conjoint <- function(name, dataset, rvar, evar, int = "", reverse = FALSE) {
  res <- conjoint(dataset, rvar = rvar, evar = evar, int = int, reverse = reverse)
  ml <- res$model_list[["full"]]
  tab <- ml$tab
  coeff <- ml$coeff
  out <- list(
    case = name,
    rvar = rvar,
    evar = res$evar,
    reverse = reverse,
    PW = list(
      Attributes = as.character(tab$PW$Attributes),
      Levels = as.character(tab$PW$Levels),
      PW = as.numeric(tab$PW$PW)
    ),
    IW = list(
      Attributes = as.character(tab$IW$Attributes),
      IW = as.numeric(tab$IW$IW)
    ),
    coeff = list(
      label = as.character(coeff$label),
      coefficient = as.numeric(coeff$coefficient),
      std.error = as.numeric(coeff$std.error),
      t.value = as.numeric(coeff$t.value),
      p.value = as.numeric(coeff$p.value)
    ),
    rsq = as.numeric(glance(ml$model)$r.squared),
    rsq_adj = as.numeric(glance(ml$model)$adj.r.squared),
    summary = paste(capture.output(summary(res)), collapse = "\n")
  )
  write_fixture(paste0("conjoint_", name), out)
}

dump_conjoint("mp3", mp3, "Rating", "Memory:Shape")
dump_conjoint("carpet", carpet, "ranking", "design:money_back", reverse = TRUE)
dump_conjoint("movie", movie, "Ranking", "price:food", reverse = TRUE)

# ============================================================================
# Expanded fixtures for full Radiant parity (added in the parity pass)
# ============================================================================

# Heterogeneous correlation matrix built with the SAME pairwise estimators the
# pyrsm.multivariate port uses: Pearson for numeric/numeric, polycor::polychor
# (2-step) for ordinal/ordinal, polycor::polyserial (2-step) for numeric/ordinal.
het_matrix <- function(df) {
  k <- ncol(df)
  iscat <- sapply(df, function(x) is.factor(x) || is.ordered(x) || is.character(x))
  m <- diag(k)
  for (i in 1:(k - 1)) {
    for (j in (i + 1):k) {
      xi <- df[[i]]; xj <- df[[j]]
      if (iscat[i] && iscat[j]) {
        r <- polycor::polychor(xi, xj)
      } else if (iscat[i] && !iscat[j]) {
        r <- polycor::polyserial(xj, xi)
      } else if (!iscat[i] && iscat[j]) {
        r <- polycor::polyserial(xi, xj)
      } else {
        r <- cor(xi, xj)
      }
      m[i, j] <- m[j, i] <- r
    }
  }
  dimnames(m) <- list(colnames(df), colnames(df))
  m
}

# ---- pre_factor: mixed numeric + ordinal (polyserial/polychoric) ------------

dump_pre_factor_mixed <- function(name, dataset, vars) {
  d <- get_data(dataset, vars)
  d <- as.data.frame(d)
  for (nm in names(d)) if (is.character(d[[nm]])) d[[nm]] <- factor(d[[nm]])
  cmat <- het_matrix(d)
  btest <- psych::cortest.bartlett(cmat, nrow(d))
  pre_kmo <- psych::KMO(cmat)
  pre_eigen <- eigen(cmat)$values
  pre_r2 <- (1 - (1 / diag(solve(cmat))))
  out <- list(
    case = name, vars = colnames(d), nobs = nrow(d), hcor = TRUE,
    cmat = mat_to_list(cmat),
    bartlett = list(chisq = as.numeric(btest$chisq), df = as.numeric(btest$df),
                    p.value = as.numeric(btest$p.value)),
    kmo = as.numeric(pre_kmo$MSA), msai = vec_to_list(pre_kmo$MSAi),
    eigen = as.numeric(pre_eigen),
    pre_r2 = vec_to_list(stats::setNames(pre_r2, colnames(d)))
  )
  write_fixture(paste0("pre_factor_", name), out)
}

dump_pre_factor_mixed("toothpaste_mixed", toothpaste, c("v1", "v2", "gender", "age.cat"))

# ---- full_factor: rotations + ML -------------------------------------------

dump_full_factor("shopping_2_quartimax", shopping, "v1:v6", nr_fact = 2, rotation = "quartimax")
dump_full_factor("shopping_2_oblimin", shopping, "v1:v6", nr_fact = 2, rotation = "oblimin")
dump_full_factor("shopping_2_simplimax", shopping, "v1:v6", nr_fact = 2, rotation = "simplimax")
dump_full_factor("shopping_ml2", shopping, "v1:v6", nr_fact = 2, rotation = "varimax", method = "maximum likelihood")
dump_full_factor("toothpaste_ml2", toothpaste, "v1:v6", nr_fact = 2, rotation = "varimax", method = "maximum likelihood")

# ---- hclus: mixed-type Gower ------------------------------------------------

dump_hclus("toothpaste_gower", toothpaste, c("v1", "v2", "v3", "gender"),
           distance = "gower")

# ---- kclus: K-Prototypes (mixed) -------------------------------------------

# radiant's kclus passes a tibble into clustMixType::kproto, which trips a
# tibble row-subassignment bug for this data. Replicate radiant's exact setup
# (standardize numeric, Gower-HC init centers, then kproto) using base
# data.frames so we capture the genuine clustMixType::kproto ground truth.
dump_kclus_kproto <- function(name, dataset, vars, nr_clus = 3, standardize = TRUE) {
  d <- as.data.frame(get_data(dataset, vars))
  for (nm in names(d)) if (is.character(d[[nm]])) d[[nm]] <- factor(d[[nm]])

  max_freq <- function(x) factor(names(which.max(table(x))), levels = levels(x))
  center_calc <- function(x) if (is.numeric(x)) mean(x) else max_freq(x)

  init <- hclus(d, vars, distance = "gower", method = "ward.D",
                max_cases = Inf, standardize = standardize)
  clus_var <- cutree(init$hc_out, k = nr_clus)

  x_std <- d
  if (standardize) {
    for (nm in names(x_std)) if (is.numeric(x_std[[nm]])) x_std[[nm]] <- as.vector(scale(x_std[[nm]]))
  }
  hc_cent <- do.call(rbind, lapply(sort(unique(clus_var)), function(g) {
    sub <- x_std[clus_var == g, , drop = FALSE]
    as.data.frame(lapply(sub, center_calc), stringsAsFactors = FALSE)
  }))
  rownames(hc_cent) <- NULL

  kp <- clustMixType::kproto(as.data.frame(x_std), k = hc_cent, iter.max = 500, verbose = FALSE)
  totss <- clustMixType::kproto(as.data.frame(x_std), k = 1, iter.max = 1, verbose = FALSE)$tot.withinss
  betweenss <- totss - kp$tot.withinss

  out <- list(
    case = name, vars = colnames(d), nr_clus = nr_clus, fun = "kproto",
    lambda = as.numeric(kp$lambda),
    sizes = as.integer(kp$size),
    tot.withinss = as.numeric(kp$tot.withinss),
    betweenss = as.numeric(betweenss),
    totss = as.numeric(totss),
    cluster = as.integer(kp$cluster)
  )
  write_fixture(paste0("kclus_", name), out)
}

dump_kclus_kproto("toothpaste_kproto", toothpaste, c("v1", "v2", "v3", "gender"), nr_clus = 3)
set.seed(1234)

# ---- mds: 3 dimensions ------------------------------------------------------

dump_mds("city_metric_3d", city, "from", "to", "distance", method = "metric", nr_dim = 3)

# ---- prmap: categorical preferences (hcor) + 3 dimensions -------------------

dump_prmap("retailers_3d", retailers, "retailer", "good_value:cluttered",
           pref = "segment1:segment2", nr_dim = 3)

# ---- conjoint: interactions + by-group -------------------------------------

dump_conjoint("mp3_int", mp3, "Rating", "Memory:Shape", int = "Memory:Shape")

# Fit one model per by-group directly (radiant's conjoint(by=) trips on the
# sig_stars formatter for saturated sub-models). evar passed explicitly so the
# by variable is excluded; each group keeps residual degrees of freedom.
dump_conjoint_by <- function(name, dataset, rvar, evar, by) {
  d <- as.data.frame(get_data(dataset, c(rvar, evar, by)))
  bylevs <- levels(as_factor(d[[by]]))
  groups <- list()
  for (lev in bylevs) {
    cdat <- d[as.character(d[[by]]) == lev, , drop = FALSE]
    cdat[[by]] <- NULL
    form <- as.formula(paste(rvar, "~", paste(evar, collapse = " + ")))
    model <- lm(form, data = cdat)
    co <- coef(model)
    tab <- the_table(tidy(model), cdat, evar)
    groups[[lev]] <- list(
      level = lev,
      coeff_label = names(co),
      coeff = as.numeric(co),
      PW_attr = as.character(tab$PW$Attributes),
      PW_lev = as.character(tab$PW$Levels),
      PW = as.numeric(tab$PW$PW),
      IW_attr = as.character(tab$IW$Attributes),
      IW = as.numeric(tab$IW$IW)
    )
  }
  out <- list(case = name, rvar = rvar, by = by, bylevs = bylevs, groups = groups)
  write_fixture(paste0("conjoint_", name), out)
}

dump_conjoint_by("mp3_by_radio", mp3, "Rating", c("Memory", "Size", "Price"), by = "Radio")

# ---- session info ----------------------------------------------------------

si <- capture.output(sessionInfo())
writeLines(si, file.path(OUT, "sessionInfo.txt"))
cat("done\n")
