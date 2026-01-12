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


import os
import pandas as pd


def load_data() -> pd.DataFrame:
    """
    Load the Adult Reconstruction dataset from the path specified
    by the DATA_PATH environment variable.

    Returns
    -------
    pd.DataFrame
        Loaded raw dataset.

    Raises
    ------
    RuntimeError
        If DATA_PATH is not set.
    """
    data_path = os.environ.get("DATA_PATH")
    if data_path is None:
        raise RuntimeError(
            "DATA_PATH environment variable is not set. "
            "Please set DATA_PATH to the location of adult_reconstruction.csv."
        )

    return pd.read_csv(data_path)



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
