"""
encode_features.py

This module performs ordinal encoding of categorical features as part of the
preprocessing pipeline described in the thesis.

In the Adult Reconstruction dataset, several predictors are categorical
(e.g., workclass, education, marital-status). Following the thesis methodology
and the Colab implementation, these features are transformed using
sklearn.preprocessing.OrdinalEncoder with:
- handle_unknown="use_encoded_value"
- unknown_value=-1

This approach produces a fully numeric feature matrix that is compatible with
the tree-based ensemble models evaluated in the thesis and supports consistent
downstream interpretability analysis (SHAP/LIME).

Expected usage order in this project:
1) load_and_clean.py  (replace "?" -> "Unknown")
2) feature_engineering.py (construct y and define X)
3) encode_features.py (encode categorical columns in X)
4) splitting + modelling stages
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify categorical and numeric feature columns.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (no target column).

    Returns
    -------
    categorical_cols : list[str]
        Columns with dtype 'object' (treated as categorical).
    numeric_cols : list[str]
        Remaining columns (treated as numeric).
    """
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(exclude="object").columns.tolist()
    return categorical_cols, numeric_cols

print(f"Categorical columns encoded: {'categorical_cols'}")

def fit_ordinal_encoder(X: pd.DataFrame, categorical_cols: List[str]) -> OrdinalEncoder:
    """
    Fit an OrdinalEncoder on the categorical columns.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    categorical_cols : list[str]
        Names of categorical columns to encode.

    Returns
    -------
    OrdinalEncoder
        Fitted encoder.
    """
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoder.fit(X[categorical_cols])
    return encoder


def transform_with_encoder(
    X: pd.DataFrame,
    encoder: OrdinalEncoder,
    categorical_cols: List[str],
) -> pd.DataFrame:
    """
    Transform categorical columns using a fitted encoder.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    encoder : OrdinalEncoder
        Fitted ordinal encoder.
    categorical_cols : list[str]
        Columns to transform.

    Returns
    -------
    pd.DataFrame
        Copy of X with categorical columns ordinal-encoded.
    """
    X_encoded = X.copy()
    X_encoded[categorical_cols] = encoder.transform(X_encoded[categorical_cols])
    return X_encoded


def encode_features(
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[OrdinalEncoder], List[str], List[str]]:
    """
    Identify categorical columns in X and apply ordinal encoding.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (unencoded).

    Returns
    -------
    X_encoded : pd.DataFrame
        Feature matrix with ordinal-encoded categorical columns.
    encoder : OrdinalEncoder or None
        Fitted encoder. Returns None if there are no categorical columns.
    categorical_cols : list[str]
        Names of encoded categorical columns.
    numeric_cols : list[str]
        Names of numeric columns.
    """
    categorical_cols, numeric_cols = identify_feature_types(X)

    # If there are no categorical columns, return X unchanged
    if len(categorical_cols) == 0:
        return X.copy(), None, categorical_cols, numeric_cols

    encoder = fit_ordinal_encoder(X, categorical_cols)
    X_encoded = transform_with_encoder(X, encoder, categorical_cols)

    return X_encoded, encoder, categorical_cols, numeric_cols


if __name__ == "__main__":
    print("encode_features.py: ordinal encoding for categorical features.")
    print("Use encode_features(X) after cleaning and feature/target splitting.")

