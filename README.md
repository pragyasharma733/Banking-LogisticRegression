# Bank Marketing: Predicting Term Deposit Subscriptions

Predicting whether a bank client will subscribe to a term deposit after a direct-marketing phone campaign, using logistic regression.

## Question
Which clients are most likely to say yes to a term deposit, so the bank can focus its call campaigns on them?

## Data
`banking.csv` is the UCI **Bank Marketing** dataset (Portuguese bank, 2008–2010). It has 41,188 clients and 20 features. The target `y` is 1 if the client subscribed, which is the case for **11.3%** of clients.

Features cover:
- **Client:** age, job, marital status, education, default/housing/loan status
- **Campaign:** contact type, month, day of week, number of contacts (`campaign`), days since last contact (`pdays`), previous outcome
- **Economic context:** employment variation rate, consumer price and confidence indices, Euribor 3-month rate, number employed

## Approach
1. **EDA:** missing-value check, target balance, Kendall correlation heatmap, and subscription rate by job, education, marital status, month and previous outcome.
2. **Cleaning:** merged `basic.4y`, `basic.6y` and `basic.9y` into a single `Basic` education level.
3. **Encoding:** one-hot encoded all categorical variables.
4. **Imbalance:** random undersampling of the majority class in the training set (3,200 / 3,200).
5. **Feature selection:** Recursive Feature Elimination (RFE) with logistic regression, keeping 30 features.
6. **Modelling:** a `statsmodels` Logit for coefficient significance, plus a scikit-learn logistic regression for prediction.

## Results
On a 30% held-out test set:

|  | Predicted no | Predicted yes |
|---|---|---|
| **Actual no** | 9,245 | 1,672 |
| **Actual yes** | 181 | 1,259 |

- Accuracy: 85%
- Recall on subscribers: ~87%
- Precision on subscribers: ~43%

Key EDA findings: subscription rates are much higher in months with few calls (Mar, Sep, Oct, Dec), for clients whose previous campaign succeeded, and for students and retirees.

## How to run
```bash
pip install numpy pandas scikit-learn imbalanced-learn statsmodels matplotlib seaborn
jupyter notebook logreg.ipynb
```

## Limitations and next steps
- **Data leakage:** `duration` (call length) is only known *after* the call, and the dataset authors recommend excluding it from realistic predictive models. It should be dropped.
- The notebook cells are out of order (the prediction cell shows a `NotFittedError`). Restart and run all.
- Features are not scaled, which causes the convergence warnings. A `Pipeline` with `StandardScaler` would fix this.
- One-hot columns are numbered (`job_4`, `month3`) rather than named, which makes the coefficients hard to read.
- Next: compare against a tree-based model, report ROC/PR AUC, and frame the results as "calls saved for the same number of sign-ups".
