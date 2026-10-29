"""Degradation-risk classifiers: candidates, threshold choice, explanations."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # optional, listed in the report as a candidate
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:  # pragma: no cover
    HAS_XGBOOST = False


def make_candidates(seed: int = 42, pos_weight: float = 1.0) -> dict:
    """Simple baseline first, then tree models (report section 8)."""
    candidates = {
        "logistic_regression": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("clf", RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                    random_state=seed, n_jobs=-1)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("clf", HistGradientBoostingClassifier(
                    max_depth=3, learning_rate=0.05, max_iter=120, min_samples_leaf=10,
                    l2_regularization=1.0, class_weight="balanced", random_state=seed)),
            ]
        ),
    }
    if HAS_XGBOOST:
        candidates["xgboost"] = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("clf", XGBClassifier(
                    n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss",
                    random_state=seed, n_jobs=2)),
            ]
        )
    return candidates


def choose_threshold(y_val, score_val) -> float:
    """Alert threshold that maximises F1 on the *validation* windows only."""
    y_val = np.asarray(y_val)
    s = np.asarray(score_val, dtype=float)
    if y_val.sum() == 0 or len(np.unique(s)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y_val, s)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def occlusion_contributions(estimator, X: pd.DataFrame, reference_values: dict):
    """Model-agnostic local explanation.

    For each monitoring signal, replace it by its typical value from healthy
    training windows and measure how much the risk score drops. A positive value
    means the signal pushes the risk *up*.
    """
    base = estimator.predict_proba(X)[:, 1]
    contrib = pd.DataFrame(0.0, index=X.index, columns=X.columns)
    for c in X.columns:
        Xm = X.copy()
        Xm[c] = reference_values[c]
        contrib[c] = base - estimator.predict_proba(Xm)[:, 1]
    return base, contrib


@dataclass
class RiskModel:
    """Fitted degradation-risk model plus everything needed to use it."""

    name: str
    estimator: object
    features: list
    threshold: float
    reference_values: dict
    f1_ref: float
    horizon: int
    drop_threshold: float
    investigate_ratio: float = 0.5
    meta: dict = field(default_factory=dict)

    def score(self, windows: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(windows[self.features])[:, 1]

    def status(self, risk) -> np.ndarray:
        risk = np.asarray(risk, dtype=float)
        return np.where(risk >= self.threshold, "high_risk",
                        np.where(risk >= self.threshold * self.investigate_ratio, "investigate", "normal"))

    def contributions(self, windows: pd.DataFrame) -> pd.DataFrame:
        _, contrib = occlusion_contributions(self.estimator, windows[self.features], self.reference_values)
        return contrib
