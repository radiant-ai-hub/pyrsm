## Uplift (Kaggle)

### Description

Data on 1,000 customers from an uplift modeling exercise posted on Kaggle. Half of the customers (500) were randomly assigned to receive an ad (the treatment group, `treat` = 1) and half were not (the control group, `treat` = 0). The response `y` records whether the customer purchased.

The data are stacked: treatment and control customers are in the same dataset, and `treat` says which group each customer was in. The features `X1` to `X20` are masked and standardized (mean about 0, standard deviation about 1); they have no known business meaning but can be used for prediction.

The goal is to find the customers whose purchase probability increases most when they are shown the ad (the uplift), rather than the customers who are most likely to purchase anyway. A typical analysis estimates one model on the treatment group and one on the control group, scores every customer with both, uses the difference as the uplift score, and evaluates it on test rows with Model > Evaluate classification > Uplift.

### Variables

* y = `yes` if the customer purchased and `no` if they did not
* treat = 1 if the customer was shown the ad (treatment), 0 if not (control)
* X1 to X20 = masked numeric features (standardized)

### Source

Posted on Kaggle as practice data for uplift modeling. The original source and the meaning of the features were not published.
