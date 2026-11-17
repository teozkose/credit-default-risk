"""Model factories: prior baseline, WoE scorecard, regularised logistic regression, tuned XGBoost."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from credit.evaluate import auc
from credit.features import CATEGORICAL
from credit.woe import WoEScorecard

MODEL_ORDER = ["baseline", "scorecard", "logistic", "xgboost", "xgboost_isotonic"]

SEARCH_SPACE: dict[str, list[Any]] = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "min_child_weight": [1, 5, 10, 20],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [1.0, 3.0, 10.0],
}


class PriorBaseline:
    """Predicts the training default rate for everyone (equivalent to 'always predict no default')."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> PriorBaseline:
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])


def make_scorecard(cfg: dict[str, Any]) -> WoEScorecard:
    w = cfg["models"]["woe"]
    return WoEScorecard(max_bins=w["max_bins"], min_bin_share=w["min_bin_share"], min_iv=w["min_iv"], C=w["C"])


def make_logistic(cfg: dict[str, Any], features: list[str]) -> Pipeline:
    """Standardised numeric features plus one-hot education, L2-penalised."""
    numeric = [f for f in features if f not in CATEGORICAL]
    pre = ColumnTransformer([("num", StandardScaler(), numeric),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=cfg["models"]["logistic"]["C"], max_iter=3000))])


def make_xgboost(cfg: dict[str, Any], **overrides: Any) -> XGBClassifier:
    params = {**cfg["models"]["xgboost"], **overrides}
    return XGBClassifier(**params, tree_method="hist", eval_metric="logloss", n_jobs=-1,
                         random_state=cfg["validation"]["seed"])


def tune_xgboost(X: pd.DataFrame, y: np.ndarray, cfg: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Small random search scored by stratified inner-CV ROC-AUC on the training fold only."""
    t = cfg["models"]["tuning"]
    seed = cfg["validation"]["seed"]
    draws = list(ParameterSampler(SEARCH_SPACE, n_iter=t["n_iter"], random_state=seed))
    inner = StratifiedKFold(n_splits=t["folds"], shuffle=True, random_state=seed)
    best, best_auc = {}, -np.inf
    for params in draws:
        scores = []
        for tr, va in inner.split(X, y):
            m = make_xgboost(cfg, **params).fit(X.iloc[tr], y[tr])
            scores.append(auc(y[va], m.predict_proba(X.iloc[va])[:, 1]))
        mean = float(np.mean(scores))
        if mean > best_auc:
            best, best_auc = params, mean
    return best, best_auc


def make_isotonic(cfg: dict[str, Any], params: dict[str, Any]) -> CalibratedClassifierCV:
    """XGBoost with an isotonic map fitted on held-out inner folds (never on its own training rows)."""
    return CalibratedClassifierCV(make_xgboost(cfg, **params), method="isotonic",
                                  cv=cfg["models"]["calibration_folds"])
