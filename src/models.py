"""
Model Definitions for Encrypted Traffic Classification
  - XGBoost (primary, cost-sensitive)
  - Random Forest
  - LightGBM
  - Voting Ensemble (RF + XGB + LGBM)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
from collections import Counter


# ── Helper: compute scale_pos_weight for binary XGBoost ─────────────────────
def _scale_pos_weight(y: np.ndarray) -> float:
    cnt = Counter(y)
    neg = cnt.get(0, 1)
    pos = cnt.get(1, 1)
    return neg / pos


# ── XGBoost (paper's primary model) ─────────────────────────────────────────
def build_xgboost(y_train: np.ndarray,
                  binary: bool = True,
                  n_estimators: int = 500,
                  max_depth: int = 8,
                  learning_rate: float = 0.05,
                  subsample: float = 0.8,
                  colsample_bytree: float = 0.8,
                  reg_alpha: float = 0.1,
                  reg_lambda: float = 1.0,
                  random_state: int = 42,
                  use_gpu: bool = False) -> xgb.XGBClassifier:
    """
    Cost-sensitive XGBoost classifier.
    scale_pos_weight is auto-set for binary; for multiclass uses sample_weight.
    """
    device = "cuda" if use_gpu else "cpu"

    common = dict(
        n_estimators    = n_estimators,
        max_depth       = max_depth,
        learning_rate   = learning_rate,
        subsample       = subsample,
        colsample_bytree= colsample_bytree,
        reg_alpha       = reg_alpha,
        reg_lambda      = reg_lambda,
        random_state    = random_state,
        n_jobs          = -1,
        device          = device,
        eval_metric     = "aucpr",
    )

    if binary:
        spw = _scale_pos_weight(y_train)
        print(f"  XGBoost scale_pos_weight = {spw:.2f}")
        return xgb.XGBClassifier(
            objective        = "binary:logistic",
            scale_pos_weight = spw,
            **common
        )
    else:
        num_classes = len(np.unique(y_train))
        return xgb.XGBClassifier(
            objective   = "multi:softprob",
            num_class   = num_classes,
            **common
        )


# ── Random Forest ────────────────────────────────────────────────────────────
def build_random_forest(y_train: np.ndarray,
                        n_estimators: int = 300,
                        max_depth: int = None,
                        max_features: str = "sqrt",
                        random_state: int = 42) -> RandomForestClassifier:
    """Class-weight balanced Random Forest."""
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict = dict(zip(classes, weights))

    return RandomForestClassifier(
        n_estimators  = n_estimators,
        max_depth     = max_depth,
        max_features  = max_features,
        class_weight  = cw_dict,
        random_state  = random_state,
        n_jobs        = -1,
    )


# ── LightGBM ─────────────────────────────────────────────────────────────────
def build_lightgbm(y_train: np.ndarray,
                   binary: bool = True,
                   n_estimators: int = 500,
                   max_depth: int = 8,
                   learning_rate: float = 0.05,
                   num_leaves: int = 63,
                   random_state: int = 42) -> lgb.LGBMClassifier:
    """LightGBM with class-weight balancing."""
    objective = "binary" if binary else "multiclass"
    metric    = "binary_logloss" if binary else "multi_logloss"
    num_class = 1 if binary else len(np.unique(y_train))

    params = dict(
        n_estimators  = n_estimators,
        max_depth     = max_depth,
        learning_rate = learning_rate,
        num_leaves    = num_leaves,
        objective     = objective,
        metric        = metric,
        class_weight  = "balanced",
        random_state  = random_state,
        n_jobs        = -1,
        verbose       = -1,
    )
    if not binary:
        params["num_class"] = num_class
    return lgb.LGBMClassifier(**params)


# ── Voting Ensemble ───────────────────────────────────────────────────────────
def build_voting_ensemble(y_train: np.ndarray,
                          binary: bool = True,
                          random_state: int = 42) -> VotingClassifier:
    """
    Soft-voting ensemble of XGBoost + LightGBM + Random Forest.
    XGBoost is given double weight as the paper's primary model.
    """
    xgb_clf = build_xgboost(y_train, binary=binary,
                             random_state=random_state)
    lgb_clf = build_lightgbm(y_train, binary=binary,
                              random_state=random_state)
    rf_clf  = build_random_forest(y_train, random_state=random_state)

    return VotingClassifier(
        estimators = [
            ("xgboost",  xgb_clf),
            ("lightgbm", lgb_clf),
            ("rf",       rf_clf),
        ],
        voting  = "soft",
        weights = [2, 1, 1],
        n_jobs  = 1,
    )


# ── Model registry ────────────────────────────────────────────────────────────
def get_model(name: str, y_train: np.ndarray,
              binary: bool = True, random_state: int = 42):
    """
    Factory function.
    name : 'xgboost' | 'random_forest' | 'lightgbm' | 'ensemble'
    """
    registry = {
        "xgboost":      lambda: build_xgboost(y_train, binary, random_state=random_state),
        "random_forest":lambda: build_random_forest(y_train, random_state=random_state),
        "lightgbm":     lambda: build_lightgbm(y_train, binary, random_state=random_state),
        "ensemble":     lambda: build_voting_ensemble(y_train, binary, random_state=random_state),
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(registry.keys())}")
    return registry[name]()


# ── Sample-weight helper for multiclass ──────────────────────────────────────
def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Return per-sample weights inversely proportional to class frequency."""
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    cw_dict = dict(zip(classes, weights))
    return np.array([cw_dict[yi] for yi in y])
