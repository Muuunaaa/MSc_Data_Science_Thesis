"""
hyperparameter_search.py

Stage 4 of the thesis pipeline: model training and hyperparameter tuning.

This module implements the baseline training and tuning workflow used in the
Colab notebook and described in the thesis (Section 4.5 / 4.5.1):

- Models: RandomForest, HistGradientBoosting (GBDT), XGBoost
- Hyperparameter tuning: RandomizedSearchCV with 3-fold cross-validation
- Scoring metric during tuning: F1-score
- Evaluation metrics: Accuracy, F1-score, ROC-AUC

Design notes (to mirror implementation):
- RandomizedSearchCV is used with n_iter=15, cv=3, scoring="f1",
  random_state=42, n_jobs=-1, verbose=1.
- The returned estimator is the fitted best_estimator_ from the search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from xgboost import XGBClassifier


RANDOM_STATE: int = 42


@dataclass(frozen=True)
class TuningResult:
    """Stores tuning output for documentation and reproducibility."""
    best_estimator: Any
    best_params: Dict[str, Any]
    best_cv_score: float


def build_base_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Create the three tree-based ensemble models used in the thesis.

    Returns
    -------
    dict
        Mapping from model name to an unfitted estimator.
    """
    models = {
        "RandomForest": RandomForestClassifier(
            n_jobs=-1,
            random_state=random_state,
        ),
        "GBDT": HistGradientBoostingClassifier(
            random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        ),
    }
    return models


def get_param_distributions() -> Dict[str, Dict[str, list]]:
    """
    Define the hyperparameter search spaces used in the Colab implementation.

    Returns
    -------
    dict
        Mapping from model name to RandomizedSearchCV param_distributions.
    """
    param_dists = {
        "RandomForest": {
            "n_estimators": [200, 300, 400],
            "max_depth": [None, 10, 20],
            "max_features": ["sqrt", "log2"],
            "min_samples_split": [2, 5, 10],
        },
        "GBDT": {
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "min_samples_leaf": [20, 50, 100],
            "max_iter": [200, 300, 400],
        },
        "XGBoost": {
            "n_estimators": [200, 300, 400],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0],
        },
    }
    return param_dists


def tune_model(
    model: Any,
    param_distributions: Dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_iter: int = 15,
    cv: int = 3,
    scoring: str = "f1",
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
    verbose: int = 1,
) -> TuningResult:
    """
    Tune a single model using RandomizedSearchCV with cross-validation.

    This mirrors the tuning strategy described in the thesis and implemented in Colab:
    RandomizedSearchCV with 3-fold CV, n_iter=15, and F1 scoring.

    Returns
    -------
    TuningResult
        best_estimator is already fitted (RandomizedSearchCV refits by default).
    """
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
    )
    search.fit(X_train, y_train)

    return TuningResult(
        best_estimator=search.best_estimator_,
        best_params=dict(search.best_params_),
        best_cv_score=float(search.best_score_),
    )


def _predict_proba_for_roc_auc(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Compute probability-like scores for ROC-AUC.

    Preference order (mirrors Colab logic):
    1) predict_proba (use positive class)
    2) decision_function (min-max scaled to [0,1])
    3) fallback to hard predictions (least desirable)
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        denom = (scores.max() - scores.min())
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
    Evaluate a fitted classifier on a given split using Accuracy, F1, ROC-AUC.

    Returns
    -------
    dict
        Keys: model, split, accuracy, f1, roc_auc
    """
    y_pred = model.predict(X)
    y_prob = _predict_proba_for_roc_auc(model, X)

    return {
        "model": model_name,
        "split": split_name,
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
    }


def train_tune_evaluate(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    n_iter: int = 15,
    cv: int = 3,
    scoring: str = "f1",
    random_state: int = RANDOM_STATE,
) -> tuple[Any, pd.DataFrame, Dict[str, Any]]:
    """
    End-to-end helper used in the Colab notebook:
    - Build base model by name
    - Tune with RandomizedSearchCV
    - Evaluate on train/val/test
    """
    base_models = build_base_models(random_state=random_state)
    param_dists = get_param_distributions()

    if model_name not in base_models:
        raise KeyError(f"Unknown model_name '{model_name}'. Choose from {list(base_models.keys())}.")

    tuning = tune_model(
        model=base_models[model_name],
        param_distributions=param_dists[model_name],
        X_train=X_train,
        y_train=y_train,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
    )

    best_model = tuning.best_estimator

    rows = [
        evaluate_model(best_model, X_train, y_train, model_name=model_name, split_name="train"),
        evaluate_model(best_model, X_val, y_val, model_name=model_name, split_name="val"),
        evaluate_model(best_model, X_test, y_test, model_name=model_name, split_name="test"),
    ]
    results_df = pd.DataFrame(rows)

    return best_model, results_df, tuning.best_params
