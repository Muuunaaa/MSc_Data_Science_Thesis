# Baseline Predictive Performance

This folder contains the code used to evaluate baseline predictive performance
on clean (unperturbed) data, as reported in the Results chapter of the thesis.

Baseline performance refers to standard classification metrics computed on the
train, validation, and test splits prior to any robustness or interpretability
analysis.

## Files

- `evaluate_baseline.py`:
  Implements reusable evaluation functions for accuracy, F1-score, and ROC-AUC.
- `baseline_metrics.ipynb`:
  Executes baseline evaluation, aggregates results across models and splits,
  and exports tables and figures used in the thesis.

## Outputs

- `results/baseline_metrics.csv`:
  Tabular baseline performance results across train/validation/test.
- `figures/baseline_test_accuracy.png`:
  Baseline test-set accuracy per model.
- `figures/baseline_test_f1.png`:
  Baseline test-set F1-score per model.
- `figures/baseline_test_roc_auc.png`:
  Baseline test-set ROC-AUC per model.

