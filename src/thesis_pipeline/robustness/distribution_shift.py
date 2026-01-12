"""
distribution_shift.py

Robustness evaluation (SQ1): simulated distribution shift.

This module implements the simple structured shift used in the Colab notebook and
described in the thesis:

- Increase age by +5 years and clip to a plausible adult range [18, 90].
- Scale capital-gain by a factor of 1.2.

All other features remain unchanged.

This shift is a lightweight simulation of systematic dataset drift, intended to
contrast with random Gaussian noise perturbations.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def apply_simple_shift(X: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """
    Apply a simple structured distribution shift to selected numeric features.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    numeric_columns : list[str]
        List of numeric columns (used to check whether target columns exist).

    Returns
    -------
    pd.DataFrame
        Copy of X with shifted feature distributions.
    """
    X_shifted = X.copy()

    if "age" in numeric_columns and "age" in X_shifted.columns:
        X_shifted["age"] = np.clip(X_shifted["age"] + 5, 18, 90)

    if "capital-gain" in numeric_columns and "capital-gain" in X_shifted.columns:
        X_shifted["capital-gain"] = X_shifted["capital-gain"] * 1.2

    return X_shifted
