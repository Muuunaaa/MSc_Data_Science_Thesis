
### ADJUST "*"!

## Baseline figures

- `baseline_test_*`:
  Baseline predictive performance on clean test data.
- `robustness_noise_*`:
  Predictive performance under Gaussian noise perturbations.
- `robustness_shift_*`:
  Predictive performance under a simulated distribution shift (clean vs shifted).

## Robustness figures

- `robustness_noise_*.png`:
  Predictive performance under Gaussian noise perturbations applied to numeric
  features (σ ∈ {0.1, 0.5, 1.0}), evaluated on the test set.
- `robustness_shift_*.png`:
  Predictive performance under a simulated distribution shift (clean vs shifted
  test data).
