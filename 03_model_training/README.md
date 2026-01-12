# Model Training and Hyperparameter Tuning

This folder implements Stage 4 of the thesis pipeline: model training and
hyperparameter tuning. T

## Models
Three tree-based ensemble methods are evaluated:
- RandomForestClassifier
- HistGradientBoostingClassifier (GBDT)
- XGBClassifier (XGBoost)

## Hyperparameter tuning
All models are tuned using `RandomizedSearchCV` with:
- 3-fold cross-validation
- `n_iter = 15` random configurations
- scoring metric: `f1`
- `random_state = 42` for reproducibility
- `n_jobs = -1` and `verbose = 1` (as in Colab)

## Evaluation
After tuning, the fitted best model is evaluated on:
- train
- validation
- test

Metrics reported:
- Accuracy
- F1-score
- ROC-AUC

## Files
- `hyperparameter_search.py`:
  shared implementation for base models, parameter grids, tuning, and evaluation.
- `train_rf.py`, `train_gbdt.py`, `train_xgboost.py`:
  thin wrappers that call the shared tuning/evaluation function for each model.
