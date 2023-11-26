# Bank Marketing classification

## Introduction
The objective of this project is to build the most effective predictive model for bank marketers using the provided dataset. The aim is to create a model suitable for real-world deployment, with the primary goal of maximizing the success of marketing campaigns in terms of both bank deposit subscriptions and marketing effort.

## Sources
- [1] Pubblication: _Moro, S., Laureano, R., & Cortez, P. (2011). Using data mining for bank direct marketing: An application of the crisp-dm methodology._
- [2] Kaggle for the data: https://www.kaggle.com/competitions/bank-marketing-uci
- [3] Kaggle solution: https://www.kaggle.com/code/antonias/bank-campaign-prediction#final-submission


## Data
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be (or not) subscribed. The classification goal is to predict if the client will subscribe a term deposit (variable y).

Features:
- age (numeric).
- job: type of job (categorical: "admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar", "self-employed", "retired", technician", "services").
- marital: marital status (categorical: "married", "divorced", "single"; note: "divorced" means divorced or widowed).
- education (categorical: "unknown", "secondary", "primary", "tertiary").
- default: has credit in default? (binary: "yes", "no").
- balance: average yearly balance, in euros (numeric). 
- housing: has housing loan? (binary: "yes", "no").
- loan: has personal loan? (binary: "yes", "no").
- contact: contact communication type (categorical: "unknown", "telephone", "cellular"). 
- day: last contact day of the month (numeric).
- month: last contact month of year (categorical: "jan", ..., "dec").
- duration: last contact duration, in seconds (numeric).
- campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact).
- pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted).
- previous: number of contacts performed before this campaign and for this client (numeric).
- poutcome: outcome of the previous marketing campaign (categorical: "unknown", "other", "failure", "success").

Target:
- y: has the client subscribed a term deposit? (binary: "yes", "no").



## Todo
- [x] Add random forest feature importance.
- [x] Add AUROC metric.
- [ ] Implement cross-validation with hyperparameter search by randomized search.
- [ ] Fit a XGBoost model (by sklearn and also its dedicated library).
- [ ] Check performances without NA imputation.
- [ ] Check if by adding more binned and new features the performance increases.
