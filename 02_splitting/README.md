# Stratified Train/Validation/Test Split

This folder implements Stage 3 of the thesis pipeline: stratified data splitting.

The split procedure is derived directly from the Methodology section on data
splitting and was checked against the Colab implementation to ensure full
consistency between thesis text and code.

## Split design

- 70% training set
- 15% validation set
- 15% test set
- stratified by the binary target (`high_income`) to preserve class proportions
- fixed random seed (`random_state = 42`) to ensure reproducibility

## File

- `split_data.py`:
  provides `stratified_train_val_test_split(X, y)` which returns
  `(X_train, X_val, X_test, y_train, y_val, y_test)`.

## Intended usage order (pipeline)

1) `01_preprocessing/load_and_clean.py`
2) `01_preprocessing/feature_engineering.py`
3) `01_preprocessing/encode_features.py`
4) `02_splitting/split_data.py`
