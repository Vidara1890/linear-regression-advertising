# Linear Regression - Advertising Sales Prediction

A simple, foundational regression project: predicting product **Sales** from
advertising spend across **TV**, **Radio**, and **Newspaper** channels, using
the classic Advertising dataset (200 markets).

This is the first in a series of small projects building up ML fundamentals
before moving on to larger end-to-end work.

## Dataset

200 rows, 4 numeric columns: `TV`, `Radio`, `Newspaper` (ad spend, $ thousands)
and `Sales` (units, thousands). No missing values.

## Approach

1. **Data exploration** — checked shape, types, missing values, distributions.
2. **Correlation analysis** — TV shows the strongest relationship with Sales,
   Radio is moderate, Newspaper is weak.
3. **Model fitting** — trained a multiple linear regression (80/20 train/test
   split) on all three channels.
4. **Evaluation** — checked residuals and test-set performance.

## Results

**Fitted coefficients:**

| Feature   | Coefficient |
|-----------|-------------|
| Intercept | 2.979       |
| TV        | 0.0447      |
| Radio     | 0.1892      |
| Newspaper | 0.0028      |

**Test set performance:**

| Metric | Value  |
|--------|--------|
| R²     | 0.899  |
| RMSE   | 1.78   |
| MAE    | 1.46   |

## Key findings

- TV and Radio spend are meaningful predictors of Sales; Newspaper spend has
  almost no measurable effect in this dataset.
- The model explains ~90% of the variance in Sales on unseen data — a strong
  fit for a simple linear model.
- Residual plots show a slight non-linear pattern at low predicted values,
  suggesting a TV×Radio interaction term could improve the fit further —
  left as future work since the goal here was a clean linear baseline.

## Project structure

```
notebooks/
  01_data_exploration.ipynb
  02_correlation_analysis.ipynb
  03_linear_regression_model.ipynb
  04_model_evaluation.ipynb
data/
outputs/
```

## Requirements

```
pandas
matplotlib
seaborn
scikit-learn
joblib
```
