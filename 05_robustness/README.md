# Robustness to Noise and Distribution Shifts (SQ1)

This folder implements the robustness evaluation described in the thesis
(Methodology §4.6 and Results §5.1).

Robustness is assessed by applying controlled perturbations to the *test*
feature set and measuring predictive performance degradation relative to the
clean test baseline. All perturbations are applied identically across models
to enable a fair comparison of robustness behaviour.

## Perturbation types

### 1) Gaussian noise (numeric features only)

Gaussian noise is added to numeric features to simulate measurement error.
Noise is drawn from a normal distribution N(0, σ), while categorical features
remain unchanged.

The evaluated noise levels follow the Colab implementation and thesis discussion:

σ ∈ {0.1, 0.5, 1.0}

Files:
- `add_noise.py`  
  Implements `add_gaussian_noise`, which perturbs numeric features only.

### 2) Simulated distribution shift (structured drift)

A simple structured distribution shift is applied to simulate dataset drift:

- `age` is increased by +5 years and clipped to the plausible adult range [18, 90]
- `capital-gain` is scaled by a factor of ×1.2

All other features remain unchanged.

Files:
- `distribution_shift.py`  
  Implements `apply_simple_shift`.

## Evaluation procedure

The notebook `robustness_evaluation.ipynb` evaluates robustness for each trained
model (Random Forest, GBDT, XGBoost) under identical perturbation conditions.
Models are *not retrained*; only the input data distribution is altered.

For each model and condition, the following metrics are computed:

- Accuracy
- F1-score
- ROC-AUC

## Outputs

### Tabular results

- `results/robustness_performance_summary.csv`  
  Summary table of predictive performance under clean, noisy, and shifted test
  conditions. This table is referenced directly in the Results chapter.

### Figures

The notebook also generates and saves robustness figures to the `figures/` folder:

- `figures/robustness_noise_*.png`  
  Performance degradation under Gaussian noise (one figure per metric).
- `figures/robustness_shift_*.png`  
  Clean vs shifted test performance comparison (one figure per metric).

These figures correspond to the robustness analyses discussed in the Results
section and are intended for direct inclusion in the thesis.
