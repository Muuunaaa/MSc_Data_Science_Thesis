"""
feature_engineering.py

This module implements Stage 2 of the methodological pipeline:
feature engineering and target construction.

Steps (aligned with the thesis and Colab workflow):
1) Create a binary target variable 'high_income' indicating whether income > 50,000 USD.
2) Define the target vector y = df['high_income'].
3) Define the feature matrix X by removing income-related columns.

Note:
- No encoding or scaling happens here. Those belong to the next preprocessing step
  (e.g., encode_features.py), to keep responsibilities clear and reproducible.
"""

import pandas as pd


def add_high_income_target(df: pd.DataFrame, income_col: str = "income", threshold: int = 50000) -> pd.DataFrame:
    """
    Add a binary target column 'high_income' to the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing an income column.
    income_col : str, default="income"
        Name of the income column in the dataset.
    threshold : int, default=50000
        Threshold for defining high income (income > threshold => 1 else 0).

    Returns
    -------
    pd.DataFrame
        Copy of the dataset with an added 'high_income' column.

    Raises
    ------
    KeyError
        If income_col is not present in the DataFrame.
    """
    if income_col not in df.columns:
        raise KeyError(f"Expected column '{income_col}' not found in dataset.")

    df_out = df.copy()
    df_out["high_income"] = (df_out[income_col] > threshold).astype(int)
    return df_out


def split_features_target(
    df: pd.DataFrame,
    income_col: str = "income",
    target_col: str = "high_income",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataset into feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset that includes both income_col and target_col.
    income_col : str, default="income"
        Income column to remove from features.
    target_col : str, default="high_income"
        Target column to predict.

    Returns
    -------
    (X, y) : tuple[pd.DataFrame, pd.Series]
        X contains all predictors, y contains the binary target.

    Raises
    ------
    KeyError
        If income_col or target_col is missing.
    """
    if income_col not in df.columns:
        raise KeyError(f"Expected column '{income_col}' not found in dataset.")
    if target_col not in df.columns:
        raise KeyError(f"Expected column '{target_col}' not found in dataset.")

    y = df[target_col]
    X = df.drop(columns=[income_col, target_col])

    return X, y


def engineer_features_and_target(
    df: pd.DataFrame,
    income_col: str = "income",
    threshold: int = 50000,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Convenience function that:
    1) creates the high_income target
    2) returns X and y
    3) also returns the augmented DataFrame (with high_income)

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    df_with_target : pd.DataFrame
        Original dataset with added 'high_income' column.
    """
    df_with_target = add_high_income_target(df, income_col=income_col, threshold=threshold)
    X, y = split_features_target(df_with_target, income_col=income_col, target_col="high_income")
    return X, y, df_with_target


if __name__ == "__main__":
    # Example usage (for reproducibility)
    # This assumes you already have a cleaned DataFrame (e.g., from load_and_clean.py)
    print("This module defines functions for target creation and X/y splitting.")
    print("Import and call engineer_features_and_target(df) in your pipeline.")
