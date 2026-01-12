# Interpretability and Subgroup Analysis (SQ2 & SQ3)

This folder contains the interpretability analyses reported in the thesis,
including global and local explanations on clean data and the robustness
of explanations under perturbations.

Interpretability is assessed using:
- SHAP (TreeSHAP) for global and local feature attributions
- LIME for local surrogate explanations

Subgroup analysis (SQ3) examines whether explanation patterns differ across:
- Gender (Male vs Female)
- Race (White vs Non-White)
- Age groups (18–30, 31–45, 46–60, 61+)

## Files

- `shap_analysis.ipynb`
  - Computes global SHAP summary plots (clean data)
  - Computes subgroup SHAP summary plots (gender, race_binary, age_group)
  - Computes SHAP robustness metrics under perturbations:
    Spearman rank correlation and L1 distance between mean-|SHAP| profiles

- `lime_analysis.ipynb`
  - Generates local LIME explanations (clean data)
  - Quantifies LIME robustness by repeating explanations per instance and
    measuring coefficient variability and feature selection variability under
    perturbations and across demographic subgroups

## Inputs (assumed)

These notebooks assume access to:
- fitted models: `best_models = {"RandomForest": ..., "GBDT": ..., "XGBoost": ...}`
- encoded train/test splits: `X_train`, `X_test`, `y_train`, `y_test`
- numeric and categorical feature lists: `numeric_cols`, `categorical_cols`
- original dataframe `df` for subgroup metadata aligned by index

## Outputs

### Results tables (saved in `results/`)
- `results/shap_robustness_summary.csv`
- `results/lime_robustness_summary.csv`

### Figures (saved in `figures/`)
- SHAP global and subgroup summary plots:
  `figures/shap_summary_clean_<MODEL>.png`,
  `figures/shap_summary_<GROUP>_<VALUE>_<MODEL>.png`
- SHAP robustness plots:
  `figures/shap_robustness_spearman.png`,
  `figures/shap_robustness_l1.png`
- LIME example explanations and robustness plots:
  `figures/lime_example_<MODEL>_clean.png`,
  `figures/lime_robustness_coeff_std.png`,
  `figures/lime_robustness_unique_features.png`
