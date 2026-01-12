"""
load_and_clean.py

This module implements the first step of the methodological pipeline:
loading the Adult Reconstruction dataset and performing minimal cleaning.

Cleaning decisions follow the methodology described in the thesis:
- Hidden missing values encoded as "?" are replaced by "Unknown"
- No rows are dropped
- No features are engineered at this stage
"""

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the Adult Reconstruction dataset from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the adult_reconstruction.csv file.

    Returns
    -------
    pd.DataFrame
        Raw dataset loaded into a DataFrame.
    """
    df = pd.read_csv(file_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform minimal cleaning of the dataset.

    Steps
    -----
    1. Replace hidden missing values ("?") with the explicit category "Unknown".
    2. Report the number of duplicate rows (duplicates are not removed).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    # Replace hidden missing values
    df = df.replace("?", "Unknown")

    # Report duplicates for transparency (do not remove)
    n_duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {n_duplicates}")

    return df


def load_and_clean(file_path: str) -> pd.DataFrame:
    """
    Convenience function to load and clean the dataset.

    Parameters
    ----------
    file_path : str
        Path to the adult_reconstruction.csv file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset ready for feature engineering.
    """
    df = load_data(file_path)
    df = clean_data(df)
    return df


if __name__ == "__main__":
    # Example usage (for reproducibility)
    path = "adult_reconstruction.csv"
    df_clean = load_and_clean(path)

    print("Dataset loaded and cleaned.")
    print(f"Shape after cleaning: {df_clean.shape}")
