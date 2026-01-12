"""
train_xgboost.py

Train + tune the XGBoost baseline model using the thesis/Colab procedure.

Note:
- Uses XGBClassifier with objective='binary:logistic', eval_metric='logloss',
  tree_method='hist', and random_state set for reproducibility (as in Colab).
"""

from __future__ import annotations

import pandas as pd
from typing import Any, Dict, Tuple

from .hyperparameter_search import train_tune_evaluate, RANDOM_STATE


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    random_state: int = RANDOM_STATE,
) -> Tuple[Any, pd.DataFrame, Dict]:
    """
    Train, tune, and evaluate XGBoost using RandomizedSearchCV (cv=3, n_iter=15, scoring='f1').

    Returns
    -------
    best_model : fitted estimator
    results_df : pd.DataFrame with metrics on train/val/test
    best_params : dict of best hyperparameters found
    """
    return train_tune_evaluate(
        model_name="XGBoost",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        random_state=random_state,
    )


if __name__ == "__main__":
    print("Import train_xgboost(...) and call it with your split datasets.")
