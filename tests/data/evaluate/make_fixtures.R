# Regenerates the radiant.model reference results that tests/test_evaluate.py
# compares pyrsm.model.evaluate against (run in a scratch dir with radiant.model
# installed). The inputs are then slimmed to the columns the tests use:
#   dvd.csv -> dvd_preds.csv (buy, pred_logit, pred_rnd, treat, training)
#   diamonds.csv -> diamonds_preds.csv (price, pred_lm, pred_rnd, training)
#   evalbin/confusion/uplift/evalreg.csv -> r_<name>.csv
suppressMessages({library(radiant.model)})
set.seed(1234)
data(dvd, package = "radiant.data")
dvd <- dvd[1:3000, ]
dvd$pred_logit <- predict(glm(buy == "yes" ~ coupon + purch + last, data = dvd, family = binomial), type = "response")
dvd$pred_rnd <- runif(nrow(dvd))
dvd$treat <- ifelse(runif(nrow(dvd)) > 0.5, "yes", "no")
dvd$training <- as.integer(seq_len(nrow(dvd)) <= 1000)
write.csv(dvd, "dvd.csv", row.names = FALSE)
eb <- evalbin(dvd, c("pred_logit", "pred_rnd"), "buy", lev = "yes", qnt = 10, cost = 1, margin = 6, train = "Both", data_filter = "training == 1")
write.csv(eb$dataset, "evalbin.csv", row.names = FALSE)
write.csv(eb$cnf_tab$dataset, "confusion.csv", row.names = FALSE)
up <- uplift(dvd, c("pred_logit", "pred_rnd"), "buy", lev = "yes", tvar = "treat", tlev = "yes", qnt = 10, cost = 1, margin = 6)
write.csv(up$dataset, "uplift.csv", row.names = FALSE)
data(diamonds, package = "radiant.data")
diamonds <- diamonds[1:1500, ]
diamonds$pred_lm <- predict(lm(price ~ carat + clarity, data = diamonds))
diamonds$pred_rnd <- rnorm(nrow(diamonds), mean(diamonds$price), sd(diamonds$price))
diamonds$training <- as.integer(seq_len(nrow(diamonds)) <= 1000)
write.csv(diamonds, "diamonds.csv", row.names = FALSE)
er <- evalreg(diamonds, c("pred_lm", "pred_rnd"), "price", train = "Both", data_filter = "training == 1")
write.csv(er$dat, "evalreg.csv", row.names = FALSE)
