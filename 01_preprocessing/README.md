# Preprocessing and Feature Engineering

This folder implements the first stages of the methodological pipeline described
in the thesis, covering data cleaning and feature preparation prior to model
training.

The preprocessing steps follow this order:
1. Load the Adult Reconstruction dataset and perform minimal cleaning
   (`load_and_clean.py`), including replacing hidden missing values ("?") with
   the explicit category "Unknown".
2. Construct the binary target variable (`high_income`) and define the feature
   matrix X and target vector y (`feature_engineering.py`).
3. Encode categorical features using ordinal encoding to obtain a fully numeric
   feature matrix compatible with tree-based models (`encode_features.py`).

No rows are removed during preprocessing, and no model training, data splitting,
or evaluation is performed in this folder. The output of these steps serves as
input for all subsequent experiments on performance, robustness, and
interpretability.
