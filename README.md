# Credit Card Default Prediction

Predicting whether a credit-card client will default on their next payment, using six machine-learning classifiers benchmarked **before and after tuning**, plus an ensemble that outperforms every individual model.

> Built end-to-end in **R** on the *Default of Credit Card Clients* dataset - from raw data cleaning to a production-ready ensemble.

---

## What This Project Does

A complete supervised-learning pipeline for a real-world, class-imbalanced binary classification problem (default vs. no-default). Each model is implemented from scratch, evaluated, then improved through class balancing and hyperparameter tuning.

**Pipeline**

1. **Data preparation** - dropped ID/constant columns, imputed missing values with **MICE**, treated outliers via **Winsorization**, removed 5 collinear features, and applied **z-score normalization**.
2. **Modelling** - trained six classifiers, each in two versions: a baseline and an improved version using **SMOTE** (class balancing), **cross-validated hyperparameter tuning**, and **Youden's J** threshold optimization.
3. **Evaluation** - compared all models on Accuracy, Sensitivity, Specificity, False Positive/Negative rates, Cohen's Kappa, and **AUC**, then selected the best via a soft-voting ensemble.

---

## Tech Stack

**Language:** R
**Libraries:** `caret`, `glmnet`, `class` (KNN), `naivebayes`, `rpart`, `randomForest`, `pROC`, `SMOTE`
**Techniques:** MICE imputation · Winsorization · multicollinearity analysis · SMOTE · k-fold cross-validation · ROC/AUC threshold tuning · ensemble (soft voting)

---

## Models Compared

Logistic Regression · K-Nearest Neighbors · Naïve Bayes · Decision Tree · Random Forest · **Ensemble (Random Forest + Naïve Bayes)**

## Results (after tuning)

| Model | Accuracy | Sensitivity | Specificity | Kappa | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.758 | 0.823 | 0.532 | 0.336 | 0.725 |
| KNN | 0.716 | 0.791 | 0.450 | 0.226 | 0.620 |
| Naïve Bayes | 0.775 | 0.852 | 0.501 | 0.351 | 0.741 |
| Decision Tree | 0.754 | 0.801 | 0.587 | 0.352 | 0.748 |
| Random Forest | 0.804 | 0.901 | 0.460 | 0.387 | 0.766 |
| **Ensemble (RF + NB)** | **0.793** | **0.874** | **0.506** | **0.388** | **0.767** |

**Best model: the RF + NB soft-voting ensemble** — highest AUC (0.767) with the most balanced trade-off between catching defaulters (sensitivity) and limiting false alarms (specificity), making it the most reliable choice for credit-risk screening.

---

## Skills Demonstrated

- End-to-end ML workflow: data cleaning → feature engineering → modelling → evaluation
- Handling **imbalanced data** with SMOTE and threshold optimization
- Hyperparameter tuning with cross-validation and parallel processing
- Model evaluation and selection using multiple metrics (not just accuracy)
- Ensemble learning and clear, metrics-driven interpretation of results

---
