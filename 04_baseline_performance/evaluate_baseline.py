"""
evaluate_baseline.py

This module implements baseline predictive performance evaluation on clean data.
It corresponds to the baseline results reported in the Results chapter of the thesis.

The evaluation uses standard classification metrics:
- Accuracy
- F1-score
- ROC-AUC

The module does not perform any training or tuning. It assumes models are already
fitted and focuses solely on evaluation and result aggregation.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def _get_scores_for_roc_auc(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Obtain probability-like scores for ROC-AUC computation.

    Preference order:
    1) predict_proba
    2) decision_function (scaled to [0, 1])
    3) fallback to binary predictions
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        denom = scores.max() - scores.min()
        if denom == 0:
            return np.zeros_like(scores, dtype=float)
        return (scores - scores.min()) / denom

    return model.predict(X)


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_name: str,
    split_name: str,
) -> Dict[str, Any]:
    """
    Evaluate a fitted model on a given data split.

    Returns
    -------
    dict
        Dictionary with model name, split name, and evaluation metrics.
    """
    y_pred = model.predict(X)
    y_score = _get_scores_for_roc_auc(model, X)

    return {
        "model": model_name,
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_score),
    }


def evaluate_baseline_models(
    models: Dict[str, Any],
    splits: Dict[str, Dict[str, pd.DataFrame | pd.Series]],
) -> pd.DataFrame:
    """
    Evaluate multiple trained models across train/validation/test splits.

    Parameters
    ----------
    models : dict
        Mapping from model name to fitted estimator.
    splits : dict
        Dictionary with keys 'train', 'val', 'test', each containing
        {'X': X_split, 'y': y_split}.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with one row per (model, split).
    """
    rows = []

    for model_name, model in models.items():
        for split_name, split_data in splits.items():
            rows.append(
                evaluate_model(
                    model,
                    split_data["X"],
                    split_data["y"],
                    model_name=model_name,
                    split_name=split_name,
                )
            )

    return pd.DataFrame(rows)
