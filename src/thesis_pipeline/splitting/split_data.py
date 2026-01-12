"""
split_data.py

Stage 3 of the methodological pipeline: stratified data splitting.

This module implements the split strategy described in the thesis:
- 70% train, 15% validation, 15% test
- stratified by the binary income target (high_income) to preserve class proportions

The implementation mirrors the Colab workflow: a two-step stratified split
(train vs. temp, then temp into validation/test) for exact 70/15/15 proportions.
"""

from __future__ import annotations

from dataclasses import dataclass
#from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitResult:
    """Container for dataset splits."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def stratified_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> SplitResult:
    """
    Split X and y into stratified train/validation/test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector (binary in this thesis: high_income).
    train_size : float, default=0.70
        Proportion of samples in the training set.
    val_size : float, default=0.15
        Proportion of samples in the validation set.
    test_size : float, default=0.15
        Proportion of samples in the test set.
    random_state : int, default=42
        Fixed seed to ensure reproducible splits.

    Returns
    -------
    SplitResult
        Dataclass containing X_train/X_val/X_test and y_train/y_val/y_test.

    Raises
    ------
    ValueError
        If the split proportions do not sum to 1.0 (within a small tolerance).
    """
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Split proportions must sum to 1.0, got {total:.6f} "
            f"(train={train_size}, val={val_size}, test={test_size})."
        )

    # Step 1: split off the training set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1.0 - train_size),
        stratify=y,
        random_state=random_state,
    )

    # Step 2: split the remaining data into validation and test
    # temp size is (val_size + test_size). We want val fraction within temp:
    val_fraction_of_temp = val_size / (val_size + test_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_fraction_of_temp),
        stratify=y_temp,
        random_state=random_state,
    )

    return SplitResult(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def _print_split_summary(splits: SplitResult) -> None:
    """Print split sizes and class proportions for transparency."""
    def frac_pos(y: pd.Series) -> float:
        return float(y.mean())

    print("=== Split summary ===")
    print(f"Train: {len(splits.y_train):,}  | positive rate: {frac_pos(splits.y_train):.4f}")
    print(f"Val:   {len(splits.y_val):,}  | positive rate: {frac_pos(splits.y_val):.4f}")
    print(f"Test:  {len(splits.y_test):,} | positive rate: {frac_pos(splits.y_test):.4f}")


if __name__ == "__main__":
    # This module is intended to be imported and used in the pipeline.
    # The example below demonstrates the intended call signature.
    print("split_data.py defines stratified_train_val_test_split(X, y).")
    print("Import it and call after preprocessing + encoding.")
