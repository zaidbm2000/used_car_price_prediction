# Used-Car Price Prediction — Machine Learning Project

## Project summary

This project builds a supervised regression workflow to predict advertised vehicle prices from vehicle characteristics. It uses a dataset of 402,005 vehicle listings and demonstrates data-quality review, exploratory data analysis, feature engineering, leakage-safe preprocessing, model comparison, cross-validation-based tuning, error analysis, and feature importance.

## Dataset

Place `adverts.csv` in the same directory as `used_car_price_prediction_portfolio.ipynb` before running the notebook. The source contains 12 columns, including price, mileage, make, model, condition, registration year, body type, colour, and fuel type.

The notebook uses the date-like prefix of `public_reference` to derive listing year/month, then drops the raw identifier. It treats an exact price of `9,999,999` as a likely sentinel and excludes those rows. Registration-derived ages below zero or above 80 are set to missing; this assumption is documented in the notebook and should be validated against the original data source.

## Machine-learning workflow

1. Inspect data types, missing values, duplicates, target distribution, and unusual values.  
2. Explore price differences by mileage, vehicle condition, make, and fuel type.  
3. Engineer vehicle age and listing date features.  
4. Split train/test rows before fitting preprocessing transformations.  
5. Use a scikit-learn `Pipeline` and `ColumnTransformer` for imputation, numeric scaling, and one-hot encoding.  
6. Compare a median baseline, Ridge regression, a decision tree, and a random forest.  
7. Tune a random forest using three-fold cross-validation on a subset of training data.  
8. Evaluate on a holdout test set, inspect residuals, and review feature importance.

## Example results from the default notebook run

The default configuration uses a reproducible 30,000-row modelling sample, with 24,000 rows for training and 6,000 for testing. In the included run, the tuned random forest achieved:

| Evaluation | MAE | RMSE | R² |
| :---- | ----: | ----: | ----: |
| Full holdout test set | 3,791 | 49,807 | 0.096 |
| Test rows at/below the training-set 99th price percentile | 2,647 | 5,512 | 0.806 |

Metrics are in the source dataset's price units; the CSV does not itself specify a currency field. The much larger overall RMSE and weaker overall R² are driven by a small number of very high-value listings that are difficult to predict from the available features. The segment result is a secondary diagnostic, not a replacement for the overall result.

Results may vary with software versions or configuration. To train on all cleaned rows, set `SAMPLE_SIZE = None` in the notebook; this will require more memory and compute time.

## How to run

1. Install Python and Jupyter (or use VS Code with the Jupyter extension).

2. Install the core packages:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

3. Put the notebook and `adverts.csv` in the same folder.

4. Open the notebook and run the cells from top to bottom.

## Limitations and next steps

- Most listing dates appear to be historical, with a heavy concentration in 2020\. Strong performance on this sample does not establish current-market pricing accuracy.  
- The random train/test split evaluates generalisation to similar-period listings, not future listings. A time-based holdout and newer external data would be stronger validation.  
- Rare luxury vehicles are poorly represented by the main price patterns and can dominate RMSE. Inspect errors by price band, make, age, and condition.  
- Additional fields such as trim, engine size, transmission, service history, location, seller type, and accident history could improve predictions.  
- Feature importance is model-specific and not a causal explanation. Permutation importance or SHAP on a validation sample would be useful follow-up analysis.

## Skills demonstrated

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Regression · Feature engineering · Pipeline design · Missing-data handling · One-hot encoding · Cross-validation · Randomized hyperparameter search · MAE/RMSE/R² · Residual diagnostics · Feature importance