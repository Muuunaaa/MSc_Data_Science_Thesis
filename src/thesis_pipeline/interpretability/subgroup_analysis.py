"""
subgroup_analysis.py

Utility functions for subgroup-based analysis used in the interpretability
experiments of the thesis (SQ2 and SQ3).

This module provides consistent logic for defining and extracting demographic
subgroups from the test data. The same subgroup definitions are reused across
SHAP and LIME analyses to ensure comparability of explanations.

The functions in this file do not perform model training or explanation
computation; they only define data partitions.
"""

from __future__ import annotations

from typing import Dict, Tuple
import pandas as pd


def split_by_categorical_attribute(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    attribute: str,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Split data into subgroups based on a categorical attribute.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    attribute : str
        Name of the categorical attribute used for subgrouping
        (e.g., 'gender', 'race', 'marital-status').

    Returns
    -------
    dict
        Mapping from subgroup label to (X_sub, y_sub).
    """
    if attribute not in X.columns:
        raise KeyError(f"Attribute '{attribute}' not found in X.")

    subgroups = {}

    for value in X[attribute].unique():
        mask = X[attribute] == value
        subgroups[str(value)] = (X.loc[mask], y.loc[mask])

    return subgroups


def split_by_age_groups(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    age_column: str = "age",
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Split data into age-based subgroups using fixed bins.

    The age bins follow common demographic groupings and match the
    age-group logic used in the thesis analysis.

    Age groups:
    - 18–29
    - 30–44
    - 45–59
    - 60+

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    age_column : str, default="age"
        Name of the age column.

    Returns
    -------
    dict
        Mapping from age-group label to (X_sub, y_sub).
    """
    if age_column not in X.columns:
        raise KeyError(f"Age column '{age_column}' not found in X.")

    bins = [18, 30, 45, 60, float("inf")]
    labels = ["18–29", "30–44", "45–59", "60+"]

    age_groups = pd.cut(X[age_column], bins=bins, labels=labels, right=False)

    subgroups = {}
    for label in labels:
        mask = age_groups == label
        subgroups[label] = (X.loc[mask], y.loc[mask])

    return subgroups


def get_subgroup_sizes(
    subgroups: Dict[str, Tuple[pd.DataFrame, pd.Series]]
) -> pd.DataFrame:
    """
    Compute subgroup sizes for reporting and sanity checks.

    Parameters
    ----------
    subgroups : dict
        Mapping from subgroup label to (X_sub, y_sub).

    Returns
    -------
    pd.DataFrame
        Table with subgroup labels and sample counts.
    """
    rows = []
    for name, (X_sub, _) in subgroups.items():
        rows.append(
            {
                "subgroup": name,
                "n_samples": len(X_sub),
            }
        )

    return pd.DataFrame(rows)
