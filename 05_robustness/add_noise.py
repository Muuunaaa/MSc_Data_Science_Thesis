"""
add_noise.py

Robustness evaluation (SQ1): Gaussian noise perturbations.

This module mirrors the robustness setup used in the Colab notebook and described
in the thesis robustness methodology:

- Gaussian noise is added only to numeric features to simulate measurement error.
- Categorical features remain unchanged.
- Noise is drawn from N(0, sigma) for a given sigma.

Design note:
- The Colab implementation draws noise from NumPy without an explicit seed.
  For reproducibility in scripted runs, this function optionally accepts a
  random_state. If random_state is None, it behaves like the notebook
  (non-deterministic across runs).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def add_gaussian_noise(
    X: pd.DataFrame,
    numeric_columns: List[str],
    *,
    sigma: float,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Add Gaussian noise to numeric features.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    numeric_columns : list[str]
        Column names in X to perturb (numeric features only).
    sigma : float
        Standard deviation of the Gaussian noise (N(0, sigma)).
    random_state : int or None, default=None
        If provided, ensures reproducible noise draws.

    Returns
    -------
    pd.DataFrame
        Copy of X where numeric_columns have been perturbed by Gaussian noise.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}.")

    X_noisy = X.copy()

    # Use NumPy Generator for optional reproducibility.
    rng = np.random.default_rng(random_state)
    noise = rng.normal(loc=0.0, scale=sigma, size=X_noisy[numeric_columns].shape)

    X_noisy[numeric_columns] = X_noisy[numeric_columns] + noise
    return X_noisy
